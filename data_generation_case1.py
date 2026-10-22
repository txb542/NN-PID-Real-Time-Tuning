### Repurposed from APmonitor (Hedengren)
### https://apmonitor.com/pdc/index.php/Main/StirredReactor
import numpy as np
from scipy.signal import find_peaks
import itertools
from scipy.integrate import odeint
from simple_pid import PID

# tauI can be used while searching for the critical gain (Section 5.2.4.1).
# It is large so that integral action is negligible during the search.
# TAU_I_CG = 10000

# define two tank system dynamic model
def two_tanks(x,t,ut,Cait,Ft,Cact,Ftott,Vt):
    # States (3):
    # Flowrate of correction stream (m3/min)
    Fc = x[0]
    # Conc. of tank 1 outlet (kg/m3)
    Ca1 = x[1]
    # Conc. of tank 2 outlet (kg/m3)
    Ca2 = x[2]
    # Parameters:
    # Kv - Valve gain (m3/(min*controller_out_units))
    Kv = 0.01

    # tauV - Valve constant (min)
    tauV = 0.1

    # Calculate valve derivative
    dFcdt = (Kv/tauV)*ut - (1/tauV)*Fc

    # Calculate concentration in tank 1 derivative
    dCa1dt = (Ft/Vt[0])*Cait + (Cact/Vt[0])*Fc - (Ftott/Vt[0])*Ca1

    # Calculate Concentration in tank 2 derivative
    dCa2dt = (Ftott/Vt[1])*Ca1 - (Ftott/Vt[1])*Ca2

    # Return xdot:
    xdot = np.zeros(3)
    xdot[0] = dFcdt
    xdot[1] = dCa1dt
    xdot[2] = dCa2dt
    return xdot

def step_change(F_sc, Cai_sc, Fc_sc, Cac_sc, Kp_sc, Ki_sc, Kd_sc):
    Kv_sc = 0.01

    u_sc = Fc_sc / Kv_sc
    # Volume of tank 1 and tank 2 (m3)
    V = np.array([5., 4.])
    
    # Total flowrate (m3/min)
    Ftot = F_sc + Fc_sc
    
    # Time Interval (min)
    t = np.linspace(0,50,2001)
    
    # System Parameters - Here they are disturbance variables (deviation form)
    
    # Initial Concentration of Tank 1 and of Tank 2 (kg/m3)
    Ca1_sc = Ca2_sc = (F_sc * Cai_sc + Fc_sc * Cac_sc) / Ftot
    
    # Store results for plotting
    Cai = np.ones(len(t)) * Cai_sc
    Cac = np.ones(len(t)) * Cac_sc
    F = np.ones(len(t)) * F_sc
    Fc = np.ones(len(t)) * Fc_sc
    Ca1 = np.ones(len(t)) * Ca1_sc
    Ca2 = np.ones(len(t)) * Ca2_sc
    u = np.ones(len(t)) * u_sc
    
    # Declare vector to pass
    x0 = np.empty(3)
    x0[0] = Fc_sc
    x0[1] = Ca1_sc
    x0[2] = Ca2_sc
    
    # setpoint
    sp = np.ones(len(t)) * (Ca2_sc + 1)
    u_bias = u_sc
    
    # Initialize PID controller
    pid = PID(Kp_sc,Ki_sc,Kd_sc)
    for i in range(len(t)-1):
        # if i % 500 == 0: # Optional tracking
        #     print("reached", i)
        # PID control
        pid.setpoint=sp[i] # Constantly updating setpoint
        u[i+1] = pid(Ca2[i], t[i+1]-t[i]) + u_bias # The controller output becomes the input - uco???
        ts = [t[i],t[i+1]]

        y = odeint(two_tanks,x0,ts,args=(u[i+1],Cai[i],F[i],Cac[i],Ftot,V))
        # Not rounding led to fluctuations at very small decimal places
        Fc[i+1] = round(y[-1][0], 5)
        Ca1[i+1] = round(y[-1][1], 5)
        Ca2[i+1] = round(y[-1][2], 5)
        x0[0] = Fc[i+1]
        x0[1] = Ca1[i+1]
        x0[2] = Ca2[i+1]
        Ftot = F[i+1] + Fc[i+1]
    
    # 
    data_to_save = np.vstack((t,u,Fc,Ca1,Ca2)) # vertical stack
    data_to_save = data_to_save.T # transpose data

    np.savetxt('data_gen.txt',data_to_save,delimiter=',',comments="",header='time,u,Fc,Ca1,Ca2')
    return Ca2, t

# Step response used in the critical gain search
# In Section 5.2.4.1 of my dissertation, I say to set tauI to 10,000 and tauD to 0.
# However, it is better to just set Ki to 0 and Kd to 0 directly.
def search_response(F_sr, Cai_sr, Fc_sr, Cac_sr, Kp_sr):
    return step_change(F_sr, Cai_sr, Fc_sr, Cac_sr, Kp_sr, 0, 0)

# Function for obtaining critical gain
# The first system starts from Kp = 1. Every later system starts from the critical gain of the
# previous system (prev_cg), see critical_gain_from_previous below.
def critical_gain(F_cg, Cai_cg, Fc_cg, Cac_cg, prev_cg=None):
    if prev_cg is not None:
        return critical_gain_from_previous(F_cg, Cai_cg, Fc_cg, Cac_cg, prev_cg)

    Kp_cg = 1

    fmax = -1
    lmax = 0
    #If there are no maxima in the output response curve:

    while abs(fmax - lmax) > 0.05:
        yout_cg, _ = search_response(F_cg, Cai_cg, Fc_cg, Cac_cg, Kp_cg) #Get the step response, function defined below
        if yout_cg[find_peaks(yout_cg)[0]].size == 0:
            Kp_cg = Kp_cg*2
        #If there is only one maxima in the output response curve:
        elif yout_cg[find_peaks(yout_cg)[0]].size == 1:
            Kp_cg = Kp_cg*2
        #If there are multiple maxima in the output response curve:
        else:
            fmax = yout_cg[find_peaks(yout_cg)[0]][0]
            lmax = yout_cg[find_peaks(yout_cg)[0]][-1]
            # Use bisection method to get critical gain
            crit_gain = bisector(Kp_cg, fmax, lmax, F_cg, Cai_cg, Fc_cg, Cac_cg)
            return crit_gain

# Critical gain search for every system after the first (Section 5.2.4.1):
# start 10 below the previous critical gain (20 below if it was greater than 100), reduce by 10
# until stable, then increase by 0.5 to find the point of instability, then bisect.
def critical_gain_from_previous(F_cg, Cai_cg, Fc_cg, Cac_cg, prev_cg):
    if prev_cg > 100:
        Kp_cg = prev_cg - 20
    else:
        Kp_cg = prev_cg - 10
    if Kp_cg < 1:
        Kp_cg = 1

    # A response is unstable when its last peak is larger than its first peak
    yout_cg, _ = search_response(F_cg, Cai_cg, Fc_cg, Cac_cg, Kp_cg)
    peaks = yout_cg[find_peaks(yout_cg)[0]]

    # If the starting gain is unstable, reduce it by 10 until a stable system is found
    while peaks.size >= 2 and peaks[-1] > peaks[0]:
        Kp_cg = Kp_cg - 10
        yout_cg, _ = search_response(F_cg, Cai_cg, Fc_cg, Cac_cg, Kp_cg)
        peaks = yout_cg[find_peaks(yout_cg)[0]]

    # Increase by 0.5 until the point of instability is found
    lower_bound = Kp_cg
    while not (peaks.size >= 2 and peaks[-1] > peaks[0]):
        lower_bound = Kp_cg
        Kp_cg = Kp_cg + 0.5
        yout_cg, _ = search_response(F_cg, Cai_cg, Fc_cg, Cac_cg, Kp_cg)
        peaks = yout_cg[find_peaks(yout_cg)[0]]

    return bisect_critical_gain(lower_bound, Kp_cg, peaks[0], peaks[-1], F_cg, Cai_cg, Fc_cg, Cac_cg)

def bisector(Kp_ub, first_max, last_max, F_ub, Cai_ub, Fc_ub, Cac_ub):
    lower_bound = Kp_ub
    upper_bound = 0 # For initialization, set upper bound to 0

    while upper_bound == 0:
        if first_max > last_max:
            lower_bound = Kp_ub
            Kp_ub = Kp_ub + 3
            yout_ub, _ = search_response(F_ub, Cai_ub, Fc_ub, Cac_ub, Kp_ub) #Get the step response
            first_max = yout_ub[find_peaks(yout_ub)[0]][0]
            last_max = yout_ub[find_peaks(yout_ub)[0]][-1]
        elif first_max < last_max:
            upper_bound = Kp_ub
        elif first_max == last_max:
            return Kp_ub

    return bisect_critical_gain(lower_bound, upper_bound, first_max, last_max, F_ub, Cai_ub, Fc_ub, Cac_ub)

### Set tolerance for having determined critical gain
def bisect_critical_gain(lower_bound, upper_bound, first_max, last_max, F_bs, Cai_bs, Fc_bs, Cac_bs):
    Kp_bs = upper_bound
    while abs(first_max - last_max) > 0.05:
        Kp_bs = (upper_bound + lower_bound) / 2
        yout_bs, _ = search_response(F_bs, Cai_bs, Fc_bs, Cac_bs, Kp_bs) #Get the step response
        first_max = yout_bs[find_peaks(yout_bs)[0]][0]
        last_max = yout_bs[find_peaks(yout_bs)[0]][-1]
        if first_max > last_max:
            lower_bound = Kp_bs
        elif first_max < last_max:
            upper_bound = Kp_bs

    return Kp_bs

# Calculate ultimate period
def ultimate_period(cg_up, F_up, Cai_up, Fc_up, Cac_up):
    Kp_up = cg_up
    yout_up, t_up_repeat = search_response(F_up, Cai_up, Fc_up, Cac_up, Kp_up) #Get the step response
    peak1 = t_up_repeat[find_peaks(yout_up)[0]][0]
    peak2 = t_up_repeat[find_peaks(yout_up)[0]][1]
    p_ult = round(peak2 - peak1, 5)
    return p_ult

# Calculate Tyreus-Luyben parameters
# prev_cg is the critical gain of the previous system (None for the first system)
def tyreus_luyben(F_tl, Cai_tl, Fc_tl, Cac_tl, prev_cg=None):
    cg_tl = critical_gain(F_tl, Cai_tl, Fc_tl, Cac_tl, prev_cg)
    print("critical gain:",cg_tl)
    ult_per = ultimate_period(cg_tl, F_tl, Cai_tl, Fc_tl, Cac_tl)
    kc_tl = round(0.45 * cg_tl, 5)
    tau_I_tl = round(2.2 * ult_per, 5)
    tau_D_tl = round(ult_per / 6.3, 5)
    return kc_tl, tau_I_tl, tau_D_tl, cg_tl

# Bring overshoot below 8% (the process specification, Section 5.2.4.1) by heuristically adjusting tauI and tauD
def check_overshoot(F_os, Cai_os, Fc_os, Cac_os, Kp_os, tauI_os, tauD_os):
    Ki_os = Kp_os/tauI_os #Calculate Ki from tauI
    Kd_os = Kp_os*tauD_os #Calculate Kd from tauD

    yout_os, t_os_repeat = step_change(F_os, Cai_os, Fc_os, Cac_os, Kp_os, Ki_os, Kd_os) #Get the step response

    Ca2_oss = (F_os * Cai_os + Fc_os * Cac_os) / (F_os + Fc_os)

    cp_os = yout_os[find_peaks(yout_os)[0]][0]

    new_Kd_os = Kd_os
    new_tauI_os = tauI_os
    add_tauI_os = 3
    sub_tauI_os = 0.05
    new_tauD_os = tauD_os
    add_tauD_os = 0.05
    sub_tauD_os = 0.05
    while cp_os >= Ca2_oss + 1.08:
        check1 = False
        check2 = False
        new_Ki_os = Kp_os/(new_tauI_os + add_tauI_os)
        new_yout_os, _ = step_change(F_os, Cai_os, Fc_os, Cac_os, Kp_os, new_Ki_os, new_Kd_os) #Get the step response
        cpIos = new_yout_os[find_peaks(new_yout_os)[0]][0]
        if cpIos < cp_os:
            cp_os = cpIos
            new_tauI_os += add_tauI_os
        elif cpIos > cp_os:
            add_tauI_os = add_tauI_os / 2
            new_Ki_os = Kp_os/(new_tauI_os - sub_tauI_os)
            new_yout_os, _ = step_change(F_os, Cai_os, Fc_os, Cac_os, Kp_os, new_Ki_os, new_Kd_os) #Get the step response
            new_cpIos = new_yout_os[find_peaks(new_yout_os)[0]][0]
            if new_cpIos < cp_os:
                cp_os = new_cpIos
                new_tauI_os -= add_tauI_os
            else:
                check1 = True
        else:
            check1 = True

        new_Kd_os = Kp_os*(new_tauD_os + add_tauD_os)
        new_yout_os, _ = step_change(F_os, Cai_os, Fc_os, Cac_os, Kp_os, new_Ki_os, new_Kd_os) #Get the step response
        cpDos = new_yout_os[find_peaks(new_yout_os)[0]][0]
        if cpDos < cp_os:
            cp_os = cpDos
            new_tauD_os += add_tauD_os
        elif cpDos > cp_os:
            new_Kd_os = Kp_os*(new_tauD_os - sub_tauD_os)
            new_yout_os, _ = step_change(F_os, Cai_os, Fc_os, Cac_os, Kp_os, new_Ki_os, new_Kd_os) #Get the step response
            new_cpDos = new_yout_os[find_peaks(new_yout_os)[0]][0]
            if new_cpDos < cp_os:
                cp_os = new_cpDos
                new_tauD_os -= sub_tauD_os
            else:
                check2 = True
        else:
            check2 = True
        if check1 == True and check2 == True:
            print("check1 and check2 both failed")
            string_to_add_os = str(F_os) + ',' + str(Cai_os) + ',' + str(Fc_os) + ',' + str(Cac_os) + ',' + str(Kp_os) + ',' + str(new_tauI_os) + ',' + str(new_tauD_os) + ',' + str(cp_os)
            f_os = open('failure_os_list.txt','a')
            f_os.write(string_to_add_os + '\n')
            f_os.close()
            break
    return round(new_tauI_os, 5), round(new_tauD_os, 5), round(Ca2_oss, 5), round(cp_os, 5)

# Check settling time. This can be added as a process spec, if desired.
# For now, it just checks and records systems which have a settling time is larger than two time constants for a given system.
def check_settling_time(F_st, Cai_st, Fc_st, Cac_st, Kp_st, tauI_st, tauD_st):
    Ki_st = Kp_st/tauI_st #Calculate Ki from tauI
    Kd_st = Kp_st*tauD_st #Calculate Kd from tauD

    V_st = np.empty(2)
    V_st[0] = 5
    V_st[1] = 4

    yout_st, t_st = step_change(F_st, Cai_st, Fc_st, Cac_st, Kp_st, Ki_st, Kd_st) #Get the step response
    #So we just need to see if what follows after the time constants is within the 2.5% of the setpoint.
    #Note that int truncates rather than rounds
    cp_st = int((V_st[0] + V_st[1]) / ((F_st + Fc_st) /0.025)) # /0.025 because time array goes in 0.025 minute increments.
    for i in range(cp_st): #This loops checks to make sure that for one time constant after the checkpoint, the output never goes outside the range
        if abs(1-yout_st[i+2*cp_st]) >= 0.025: #2.5 margin of error on either side
            string_to_add_st = str(F_st) + ',' + str(Cai_st) + ',' + str(Fc_st) + ',' + str(Cac_st) + ',' + str(Kp_st) + ',' + str(tauI_st) + ',' + str(tauD_st) + ',' + str(t_st[i])
            f_st = open('failure_st_list.txt','a')
            f_st.write(string_to_add_st + '\n')
            f_st.close()
            break
        else:
            pass

#   def append_row(path, *values, mode='a'):
#       with open(path, mode) as f:
#           f.write(','.join(map(str, values)) + '\n')

# Write headers for OS and ST failure lists
f1 = open('failure_st_list.txt','w')
f1.write('F,Cai,Fc,Cac,Kp,tauI,tauD,t_st[i]\n')
f1.close()
f2 = open('failure_os_list.txt','w')
f2.write('F,Cai,Fc,Cac,Kp,tauI,tauD,ca2_ss,cp_os\n')
f2.close()

# These ranges will generate 104,181 samples
F_list = np. array([0.5,1,2,3,4,5,6,7,8,9,10])
Cai_list = np.arange(5,15.1,0.5)
Fc_list = np.arange(0,0.51,0.05)
Cac_list = np.arange(100,301,5)

# Save header row in CSV
f = open('training_data_case1.csv','w')
f.write('F,Cai,Fc,Cac,Kp,tauI,tauD,Ca2_ss,overshoot\n')
f.close()

# Generate training data
prev_cg = None # The first system has no previous critical gain, so its search starts from Kp = 1
for w, x, y, z in itertools.product(range(len(F_list)), range(len(Cai_list)), range(len(Fc_list)), range(len(Cac_list))):
    print(F_list[w], Cai_list[x], Fc_list[y], Cac_list[z])
    tl = tyreus_luyben(F_list[w], Cai_list[x], Fc_list[y], Cac_list[z], prev_cg)
    prev_cg = tl[3] # Critical gain of this system, used to start the search for the next one
    print(tl)
    newtauI, newtauD, Ca2ss, osv = check_overshoot(F_list[w], Cai_list[x], Fc_list[y], Cac_list[z], tl[0], tl[1], tl[2])
    check_settling_time(F_list[w], Cai_list[x], Fc_list[y], Cac_list[z], tl[0], newtauI, newtauD)
    string_to_add = str(F_list[w]) + ',' + str(Cai_list[x]) + ',' + str(Fc_list[y]) + ',' + str(Cac_list[z]) + ',' + str(tl[0]) + ',' + str(newtauI) + ',' + str(newtauD) + ',' + str(Ca2ss) + ',' + str(osv)
    f = open('training_data_case1.csv','a')
    f.write(string_to_add + '\n')
    f.close()

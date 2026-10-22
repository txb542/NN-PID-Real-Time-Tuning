import numpy as np
from scipy.signal import find_peaks
import itertools
from scipy.integrate import odeint
from simple_pid import PID

# tauI can be used while searching for the critical gain (Section 5.2.4.1).
# It is large so that integral action is negligible during the search.
# TAU_I_CG = 10000

# define two tank system model using simplified model
def two_tanks(x,t,u,Cai,tau1,tau2,Cac_F):
    # Inputs (2):
    # Redefine controller variable
    uco = u

    # Cai will be taken from what is passed
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
    dFcdt = (Kv/tauV)*uco - (1/tauV)*Fc

    # Calculate concentration in tank 1 derivative
    dCa1dt = (1/tau1)*Cai + (Cac_F/tau1)*Fc - (1/tau1)*Ca1

    # Calculate Concentration in tank 2 derivative
    dCa2dt = (1/tau2)*Ca1 - (1/tau2)*Ca2

    # Return xdot:
    xdot = np.zeros(3)
    xdot[0] = dFcdt
    xdot[1] = dCa1dt
    xdot[2] = dCa2dt
    return xdot

def step_change(Kp_s, Ki_s, Kd_s, tau1_sc, tau2_sc, Cac_F_sc):
    # Steady state initial conditions in deviation form
    Fc_ss = 0
    Ca1_ss = 0
    Ca2_ss = 0
    x0 = np.empty(3)
    x0[0] = Fc_ss
    x0[1] = Ca1_ss
    x0[2] = Ca2_ss
    # Steady State Initial Condition
    u_ss = 0
    # Disturbance Variables
    # Inlet Concentration (deviation form) (kg/m^3)
    Cai = 0
    # Time Interval (min)
    if Cac_F_sc >= 1000:
        t = np.linspace(0,75,3001)
    else:
        t = np.linspace(0,100,4001)
    # Store results for plotting
    Fc = np.ones(len(t)) * Fc_ss
    Ca1 = np.ones(len(t)) * Ca1_ss
    Ca2 = np.ones(len(t)) * Ca2_ss
    u = np.ones(len(t)) * u_ss
    # setpoint
    sp = np.ones(len(t)) # * Ca_ss
    u_bias = u_ss
    # Initialize PID controller
    pid = PID(Kp_s,Ki_s,Kd_s)
    # Simulate tank system
    for i in range(len(t)-1):
        # PID control
        pid.setpoint=sp[i] # Constantly updating setpoint
        u[i+1] = pid(Ca2[i], t[i+1]-t[i]) + u_bias # The controller output becomes the input to the system
        ts = [t[i],t[i+1]]
        y = odeint(two_tanks,x0,ts,args=(u[i+1],Cai,tau1_sc,tau2_sc,Cac_F_sc))
        Fc[i+1] = y[-1][0]
        Ca1[i+1] = y[-1][1]
        Ca2[i+1] = y[-1][2]
        x0[0] = Fc[i+1]
        x0[1] = Ca1[i+1]
        x0[2] = Ca2[i+1]

    return Ca2, t

# Step response used in the critical gain search
# In Section 5.2.4.1 of my dissertation, I say to set tauI to 10,000 and tauD to 0.
# However, it is better to just set Ki to 0 and Kd to 0 directly.
def search_response(tau1_sr, tau2_sr, Cac_F_sr, Kp_sr):
    return step_change(Kp_sr, 0, 0, tau1_sr, tau2_sr, Cac_F_sr)

def critical_gain(tau1_cg, tau2_cg, Cac_F_cg):
    Kp_cg = 1

    fmax = -1
    lmax = 0
    # If there are no maxima in the output response curve:
    while abs(fmax - lmax) > 0.05:
        yout_cg, _ = search_response(tau1_cg, tau2_cg, Cac_F_cg, Kp_cg) #Get the step response
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
            crit_gain = bisector(Kp_cg, fmax, lmax, tau1_cg, tau2_cg, Cac_F_cg)
            return crit_gain

def bisector(Kp_ub, first_max, last_max, tau1_ub, tau2_ub, Cac_F_ub):
    lower_bound = Kp_ub
    upper_bound = 0 # For initialization, set upper bound to 0

    while upper_bound == 0:
        if first_max - last_max > -0.0005:
            lower_bound = Kp_ub
            Kp_ub = Kp_ub + 10
            yout_ub, _ = search_response(tau1_ub, tau2_ub, Cac_F_ub, Kp_ub) #Get the step response
            first_max = yout_ub[find_peaks(yout_ub)[0]][0]
            last_max = yout_ub[find_peaks(yout_ub)[0]][-1]
        elif first_max < last_max:
            upper_bound = Kp_ub
        elif first_max == last_max:
            return Kp_ub

    return bisect_critical_gain(lower_bound, upper_bound, first_max, last_max, tau1_ub, tau2_ub, Cac_F_ub)

### Set tolerance for having obtained critical gain
def bisect_critical_gain(lower_bound, upper_bound, first_max, last_max, tau1_bcg, tau2_bcg, Cac_F_bcg):
    Kp_bcg = upper_bound
    while abs(first_max - last_max) > 0.05:
        Kp_bcg = (upper_bound + lower_bound) / 2
        yout_bcg, _ = search_response(tau1_bcg, tau2_bcg, Cac_F_bcg, Kp_bcg) #Get the step response
        first_max = yout_bcg[find_peaks(yout_bcg)[0]][0]
        last_max = yout_bcg[find_peaks(yout_bcg)[0]][-1]
        if first_max > last_max:
            lower_bound = Kp_bcg
        elif first_max < last_max:
            upper_bound = Kp_bcg

    return Kp_bcg

def ultimate_period(cg_up, tau1_up, tau2_up, Cac_F_up):
    Kp_up = cg_up
    yout_up, t_up = search_response(tau1_up, tau2_up, Cac_F_up, Kp_up) #Get the step response
    peak1 = t_up[find_peaks(yout_up)[0]][0]
    peak2 = t_up[find_peaks(yout_up)[0]][1]
    p_ult = round(peak2 - peak1, 5)
    return p_ult

def tyreus_luyben(tau1_tl, tau2_tl, Cac_F_tl):
    cg_tl = critical_gain(tau1_tl, tau2_tl, Cac_F_tl)
    ult_per = ultimate_period(cg_tl, tau1_tl, tau2_tl, Cac_F_tl)
    kc_tl = round(0.45 * cg_tl, 5)
    tau_I_tl = round(2.2 * ult_per, 5)
    tau_D_tl = round(ult_per / 6.3, 5)
    return kc_tl, tau_I_tl, tau_D_tl

def check_overshoot(tau1_os, tau2_os, Cac_F_os, Kp_os, tauI_os, tauD_os):
    Ki_os = Kp_os/tauI_os #Calculate Ki from tauI
    Kd_os = Kp_os*tauD_os #Calculate Kd from tauD

    # t_os = np.linspace(0,99.975,2000) #Define timespace
    yout_os, t_os_repeat = step_change(Kp_os, Ki_os, Kd_os, tau1_os, tau2_os, Cac_F_os) #Get the step response

    cp_os = yout_os[find_peaks(yout_os)[0]][0]

    new_Kd_os = Kd_os

    new_tauI_os = tauI_os
    add_tauI_os = 1
    sub_tauI_os = 0.05
    new_tauD_os = tauD_os
    add_tauD_os = 0.05
    sub_tauD_os = 0.05
    # Dataset posted used a tolerance of 1.05.
    while cp_os >= 1.08:
        check1 = False
        check2 = False
        new_Ki_os = Kp_os/(new_tauI_os + add_tauI_os)
        new_yout_os, new_t_os = step_change(Kp_os, new_Ki_os, new_Kd_os, tau1_os, tau2_os, Cac_F_os) #Get the step response
        cpIos = new_yout_os[find_peaks(new_yout_os)[0]][0]
        if cpIos < cp_os:
            cp_os = cpIos
            new_tauI_os += add_tauI_os
        elif cpIos > cp_os:
            new_Ki_os = Kp_os/(new_tauI_os - sub_tauI_os)
            new_yout_os, new_t_os = step_change(Kp_os, new_Ki_os, new_Kd_os, tau1_os, tau2_os, Cac_F_os) #Get the step response
            new_cpIos = new_yout_os[find_peaks(new_yout_os)[0]][0]
            if new_cpIos < cp_os:
                cp_os = new_cpIos
                new_tauI_os -= sub_tauI_os
            else:
                check1 = True
        else:
            check1 = True

        new_Kd_os = Kp_os*(new_tauD_os + add_tauD_os)
        new_yout_os, new_t_os = step_change(Kp_os, new_Ki_os, new_Kd_os, tau1_os, tau2_os, Cac_F_os) #Get the step response
        cpDos = new_yout_os[find_peaks(new_yout_os)[0]][0]
        if cpDos < cp_os:
            cp_os = cpDos
            new_tauD_os += add_tauD_os
        elif cpDos > cp_os:
            new_Kd_os = Kp_os*(new_tauD_os - sub_tauD_os)
            new_yout_os, new_t_os = step_change(Kp_os, new_Ki_os, new_Kd_os, tau1_os, tau2_os, Cac_F_os) #Get the step response
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
            string_to_add_os = str(tau1_os) + ',' + str(tau2_os) + ',' + str(Cac_F_os) + ',' + str(Kp_os) + ',' + str(new_tauI_os) + ',' + str(new_tauD_os) + ',' + str(cp_os)
            f_os = open('failure_overshoot_list.txt','a')
            f_os.write(string_to_add_os + '\n')
            f_os.close()
            break
    return round(new_tauI_os, 5), round(new_tauD_os, 5), round(cp_os, 5)

def check_settling_time(tau1_st, tau2_st, Cac_F_st, Kp_st, tauI_st, tauD_st):
    Ki_st = Kp_st/tauI_st #Calculate Ki from tauI
    Kd_st = Kp_st*tauD_st #Calculate Kd from tauD

    yout_st, t_st = step_change(Kp_st, Ki_st, Kd_st, tau1_st, tau2_st, Cac_F_st) #Get the step response
    # print("maximum value =", yout_st[find_peaks(yout_st)[0]][0])

    cp_st = int((tau1_st + tau2_st) / 0.025) # /0.025 because time array goes in 0.025 minute increments.
    for i in range(cp_st): #This loops checks to make sure that for one time constant after the checkpoint, the output never goes outside the range
        if abs(1-yout_st[i+3*cp_st]) >= 0.025: #2.5 margin of error on either side
            string_to_add_st = str(tau1_st) + ',' + str(tau2_st) + ',' + str(Cac_F_st) + ',' + str(Kp_st) + ',' + str(tauI_st) + ',' + str(tauD_st) + ',' + str(t_st[i])
            f_st = open('failure_settling_time_list.txt','a')
            f_st.write(string_to_add_st + '\n')
            f_st.close()
            break
        else:
            pass

# Write header rows for OS and ST failure lists
f1 = open('failure_settling_time_list.txt','w')
f1.write('tau1,tau2,Cac/F,Kp,tauI,tauD,t_st[i]\n')
f1.close()
f2 = open('failure_overshoot_list.txt','w')
f2.write('tau1,tau2,Cac/F,Kp,tauI,tauD,cp_os\n')
f2.close()

# Ranges set for training set (Table 12)
t1_list = np.linspace(2.4,10,20)
t2_list = np.linspace(2.4,10,20)
Cac_F_list = np.linspace(100,484,25)

# For the larger training dataset (Table 13): (MAKE SURE TO UPDATE FILE NAME)
# t1_list = np.linspace(2.4,10,45)
# t2_list = np.linspace(2.4,10,45)
# Cac_F_list = np.linspace(100,484,50)

### For testing data, generate 500 random values from within the ranges directly above

f = open('training_dataset_case2.csv','w')
f.write('tau1,tau2,Cac/F,Kp,tauI,tauD,overshoot\n')
f.close()

for x, y, z in itertools.product(range(len(t1_list)), range(len(t2_list)), range(len(Cac_F_list))):
    print(t1_list[x],t2_list[y],Cac_F_list[z])
    tl = tyreus_luyben(t1_list[x], t2_list[y], Cac_F_list[z])
    newtauI, newtauD, osv = check_overshoot(t1_list[x], t2_list[y], Cac_F_list[z], tl[0], tl[1], tl[2])
    check_settling_time(t1_list[x], t2_list[y], Cac_F_list[z], tl[0], newtauI, newtauD)
    string_to_add = str(t1_list[x]) + ',' + str(t2_list[y]) + ',' + str(Cac_F_list[z]) + ',' + str(tl[0]) + ',' + str(newtauI) + ',' + str(newtauD) + ',' + str(osv)
    f = open('training_dataset_case2.csv','a')
    f.write(string_to_add + '\n')
    f.close()

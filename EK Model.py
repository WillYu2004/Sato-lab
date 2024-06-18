import numpy as np
import matplotlib.pyplot as plt


# Define the ODEs
def derivatives(t, state, J_si):
    v, h, f = state

    minf = ((v / 0.2) ** 6) / (1 + ((v / 0.2) ** 6))
    hinf = 1 / (1 + ((v / 0.1) ** 6))
    dinf = ((v / 0.4) ** 4) / (1 + ((v / 0.4) ** 4))
    finf = 1 / (1 + ((v / 0.1) ** 4))

    tauh = tauh1 + tauh2 * np.exp(-20 * ((v - 0.1) ** 2))
    tauf = tauf2 + (tauf1 - tauf2) * v ** 3

    jfi = h * minf * (v - 1.3) / taufi
    jsi = f * dinf * (v - 1.4) / tausi * J_si
    jso = (1 - np.exp(-4 * v)) / tauso
    ion = -(jfi + jsi + jso - stim(t))

    dvdt = ion
    dhdt = (hinf - h) / tauh
    dfdt = (finf - f) / tauf

    return [dvdt, dhdt, dfdt]


# Stimulation function
def stim(t):
    if (t % pcl < 1.0):
        return 0.3
    else:
        return 0


# Runge-Kutta 4th order method
def runge_kutta(f, t0, y0, t_end, dt, J_si):
    t_values = np.arange(t0, t_end + dt, dt)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]

        k1 = np.array(f(t, y, J_si))
        k2 = np.array(f(t + dt / 2, y + dt * k1 / 2, J_si))
        k3 = np.array(f(t + dt / 2, y + dt * k2 / 2, J_si))
        k4 = np.array(f(t + dt, y + dt * k3, J_si))

        y_values[i] = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_values, y_values


# Parameters
v0 = 0.0
h0 = 1
f0 = 0.9
initial_conditions = [v0, h0, f0]

tauso = 15
taufi = 0.8
tauh1 = 4.8
tauh2 = 10.0

tausi = 4.0
tauf1 = 100
tauf2 = 30

J_si = 1
dt = 0.1
pcl = 200
itr = 10
tmax = pcl * itr

# Solve the ODE using the Runge-Kutta method
t_values, y_values = runge_kutta(derivatives, 0, initial_conditions, tmax, dt, J_si)

# Extract results
v_values = y_values[:, 0]
h_values = y_values[:, 1]
f_values = y_values[:, 2]


# Function to detect APD and DI
def detect_apd_di(t_values, v_values, threshold=0.1):
    apd_list = []
    di_list = []
    pcl_list = []

    apd_start = None
    apd_end = None
    last_apd_end = None

    for i in range(1, len(v_values)):
        if v_values[i - 1] < threshold <= v_values[i]:
            apd_start = t_values[i]
        elif v_values[i - 1] >= threshold > v_values[i]:
            apd_end = t_values[i]
            if apd_start is not None:
                apd = apd_end - apd_start
                apd_list.append(apd)
                if last_apd_end is not None:
                    di = apd_start - last_apd_end
                    di_list.append(di)
                    pcl = apd + di
                    pcl_list.append(pcl)
                last_apd_end = apd_end
            apd_start = None
            apd_end = None

    return apd_list, di_list, pcl_list


# Detect APD and DI
apd_list, di_list, pcl_list = detect_apd_di(t_values, v_values)

# Ensure lists have the same length
min_length = min(len(apd_list), len(di_list), len(pcl_list))
apd_list = apd_list[:min_length]
di_list = di_list[:min_length]
pcl_list = pcl_list[:min_length]

# Plot APD vs PCL
plt.figure()
plt.plot(pcl_list, apd_list, 'o-')
plt.title('APD vs PCL')
plt.xlabel('PCL (ms)')
plt.ylabel('APD (ms)')
plt.grid(True)
plt.show()

# Plot APD_n+1 vs DI_n
plt.figure()
plt.plot(di_list[:-1], apd_list[1:], 'o-')
plt.title('APD_{n+1} vs DI_n')
plt.xlabel('DI_n (ms)')
plt.ylabel('APD_{n+1} (ms)')
plt.grid(True)
plt.show()

# Plot results
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t_values, v_values)
plt.title('Echebarria-Karma model')
plt.xlabel('time (ms)')
plt.ylabel('normalized Vm')

plt.subplot(3, 1, 2)
plt.plot(t_values, h_values)
plt.xlabel('time (ms)')
plt.ylabel('h gate')

plt.subplot(3, 1, 3)
plt.plot(t_values, f_values)
plt.xlabel('time (ms)')
plt.ylabel('f gate')

plt.tight_layout()
plt.show()

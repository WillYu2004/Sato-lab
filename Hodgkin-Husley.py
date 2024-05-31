import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Set random seed (for reproducibility)
np.random.seed(1000)

# Start and end time (in milliseconds)
tmin = 0.0
tmax = 20.0

# Average potassium channel conductance per unit area (mS/cm^2)
gK = 36.0

# Average sodium channel conductance per unit area (mS/cm^2)
gNa = 120.0

# Average leak channel conductance per unit area (mS/cm^2)
gL = 0.3

# Membrane capacitance per unit area (uF/cm^2)
Cm = 1.0

# Potassium potential (mV)
VK = -12.0

# Sodium potential (mV)
VNa = 115.0

# Leak potential (mV)
Vl = 10.613

# Time values
T = np.linspace(tmin, tmax, 10000)

# Potassium ion-channel rate functions
def alpha_n(Vm):
    return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)

def beta_n(Vm):
    return 0.125 * np.exp(-Vm / 80.0)

# Sodium ion-channel rate functions
def alpha_m(Vm):
    return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)

def beta_m(Vm):
    return 4.0 * np.exp(-Vm / 18.0)

def alpha_h(Vm):
    return 0.07 * np.exp(-Vm / 20.0)

def beta_h(Vm):
    return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)

# n, m, and h steady-state values
def n_inf(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))

def m_inf(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))

def h_inf(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))

def tau_n(V):
    return 1.0 / (alpha_n(V) + beta_n(V))

def tau_m(V):
    return 1.0 / (alpha_m(V) + beta_m(V))

def tau_h(V):
    return 1.0 / (alpha_h(V) + beta_h(V))
# Input stimulus
def Id(t):
    if 0.0 < t < 1.0:
        return 150.0
    return 0.0

# Compute derivatives
def compute_derivatives(y, t0):
    dy = np.zeros((4,))

    Vm = y[0]
    n = y[1]
    m = y[2]
    h = y[3]

    # dVm/dt
    GK = (gK / Cm) * np.power(n, 4.0)
    GNa = (gNa / Cm) * np.power(m, 3.0) * h
    GL = gL / Cm

    dy[0] = (Id(t0) / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl))

    # dn/dt
    dy[1] = (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)

    # dm/dt
    dy[2] = (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)

    # dh/dt
    dy[3] = (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)

    return dy

# State (Vm, n, m, h)
Y = np.array([0.0, n_inf(), m_inf(), h_inf()])

# Solve ODE system
Vy = odeint(compute_derivatives, Y, T)

# Input stimulus
Idv = [Id(t) for t in T]

# define voltage range

V = np.linspace(0, 100, 400)

n_inf_values = n_inf(V)
m_inf_values = m_inf(V)
h_inf_values = h_inf(V)

tau_n_values = tau_n(V)
tau_m_values = tau_m(V)
tau_h_values = tau_h(V)

plt.subplot(1, 2, 1)
plt.plot(V, n_inf_values, label='$n_{\infty}(V)$')
plt.plot(V, m_inf_values, label='$m_{\infty}(V)$')
plt.plot(V, h_inf_values, label='$h_{\infty}(V)$')
plt.xlabel('Potential (mV)')
plt.ylabel('Steady-state values')
plt.title('Steady-State Functions')
plt.legend()
plt.grid()

# Plot time constants
plt.subplot(1, 2, 2)
plt.plot(V, tau_n_values, label='$\\tau_n(V)$')
plt.plot(V, tau_m_values, label='$\\tau_m(V)$')
plt.plot(V, tau_h_values, label='$\\tau_h(V)$')
plt.xlabel('Potential (mV)')
plt.ylabel('Time constant (ms)')
plt.title('Time Constants')
plt.legend()
plt.grid()

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(T, Vy[:, 0], label='Vm (mV)')
plt.xlabel('Time (ms)')
plt.ylabel('Vm (mV)')
plt.title('Action Potential')
plt.legend()
plt.grid()

# Plot the gating variables (n, m, h)
plt.subplot(3, 1, 2)
plt.plot(T, Vy[:, 1], label='n')
plt.plot(T, Vy[:, 2], label='m')
plt.plot(T, Vy[:, 3], label='h')
plt.xlabel('Time (ms)')
plt.ylabel('Gating Variables')
plt.title('Gating Variables during Action Potential')
plt.legend()
plt.grid()

# Plot the conductances (gK, gNa)
GK = gK * np.power(Vy[:, 1], 4.0)
GNa = gNa * np.power(Vy[:, 2], 3.0) * Vy[:, 3]

plt.subplot(3, 1, 3)
plt.plot(T, GK, label='gK (mS/cm^2)')
plt.plot(T, GNa, label='gNa (mS/cm^2)')
plt.xlabel('Time (ms)')
plt.ylabel('Conductance (mS/cm^2)')
plt.title('Conductances during Action Potential')
plt.legend()
plt.grid()

plt.show()

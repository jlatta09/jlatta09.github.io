import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


## material constants are for Aluminum

A = 0.05 # cross sectional area of wall element in m^2 ### This is the variable of convern - the area that is exposed to radiative heat transfer ###
rho = 2700.0 # density of wall material in kg / m^3
k = 200 # thermal conductivity of wall material in W / (m*K)
c = 921.0 # specific heat capacity in J / (kg*K)
sigma = 5.6704E-08 # Stefan-Boltzmann constant in W * m^-2 * K^-4
h = 20.0 # convective heat transfer coefficient in W / (m^2 * K)
T_initial = 933.0 # initial temperature K
T_inf = 4.0 # space temperature in K
yrs = 2 # number of years to analyze temp
L = 0.5 # thickness of the entire wall in m
N = 40 # number of discrete wall segments
dx = L/N # length of each wall segment in m
eps = 0.05 # emissivity
p = 0.04 # sectional perimeter m
powDen = 0.54 # Power output in W/g of Pu
massPu = 4500
total_time = yrs*3600.0 # total duration of simulation in seconds
nsteps = 10000 # number of timesteps
dt = total_time/nsteps # duration of timestep in seconds

# x-dir discretization factor
heatfac = dx / (k*A)
# time discretization factor
simfac = (k*dt) / (c*rho*dx*dx)

x = np.linspace(0, dx*(N-1), N)
timesamps = np.linspace(0, dt*nsteps, nsteps+1)

X, TIME = np.meshgrid(x, timesamps, indexing='ij')

# initialize 2D array to store temperature values
T = np.zeros((X.shape))

nsteps = 500 # number of timesteps
dt = total_time/nsteps # duration of timestep in seconds

Theat = np.zeros((X.shape)) # Array to store Temp values

# initial temperature profile of the wall
#
# Exponential decay IC's
# for i in range(len(x)):
#    Theat[i, 0] = T_initial*np.exp(-x[i])
#
# Melting Point IC's
for i in range(len(x)):
   Theat[i, 0] = T_initial
#
# Ambient IC's
# for i in range(len(x)):
#    Theat[i, 0] = T_inf

Q_dot_in = massPu*powDen # power input from RTG

for j in range(len(timesamps)-1):

   T_out = Theat[len(x)-1, j]
   Q_dot_out = eps * sigma * A * (pow(T_out,4) - pow(T_inf,4)) # Power out from radiative cooling
   # temperature at the outside boundary
   Theat[len(x)-1, j+1] = T_out + simfac * (Theat[len(x)-2, j] - T_out - heatfac*Q_dot_out)
   # temperature at the inside boundary
   Theat[0, j+1] = Theat[0,j] + simfac * (Theat[1,j] - Theat[0,j] + heatfac * Q_dot_in - heatfac*Q_dot_out)
   # interior node temperatures
   for i in range(len(x)-2):
      Theat[i+1,j+1] = Theat[i+1,j] + simfac * (Theat[i,j] - 2*Theat[i+1,j] + Theat[i+2,j] - heatfac*Q_dot_out)

# temperature vs time data as a surface
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')

foo = ax.plot_surface(X*1000, TIME, Theat, cmap='hot')
fig1.colorbar(foo,shrink=0.5,aspect=5)

ax.set_xlabel('x (cm)')
ax.set_ylabel('time (hr)')
ax.set_zlabel('temperature (K)')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')

foo2 = ax2.plot_surface(TIME, X*1000, Theat, cmap='hot')
fig2.colorbar(foo,shrink=0.5,aspect=5)

ax2.set_xlabel('time (hr)')
ax2.set_ylabel('x (cm)')
ax2.set_zlabel('temperature (K)')

plt.show()

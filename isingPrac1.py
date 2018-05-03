import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt



N = 10 # square lattice size
shape = (N,N) # set up NxN lattice
spin = np.random.choice([-1,1], size=shape) # random spin cofiguration of -1 or 1
ExField = np.full(shape,0) # fill External magnetic field with 0's
tempStep = 20 # temperature steps
MCstep = 100 # Monte Carlo steps
T = np.linspace(0.01, 4, tempStep) # temperature range for Energy and Magnetization calcs
simTemp = 0.1 # constant temp for spin simulation

MCEnergy = np.zeros(tempStep) # initialize Energy array
MCMagnetization = np.zeros(tempStep) # initialize Magnetization array

### MCmetro utilizes the Metropolis algorithm to determine spin:
### -if flipping spin results in Energy decrease, it is flipped
### -if flipping spin results in constant or increased Energy, the probability it is
###  flipped = exp(-deltaE/kT)

def MCmetro(spin, beta):

    for i in range(N):

        for j in range(N):

                rand1 = np.random.randint(0, N)
                rand2 = np.random.randint(0, N)
                detSpin =  spin[rand1, rand2]
                met = spin[(rand1+1)%N,rand2] + spin[rand1,(rand2+1)%N] + spin[(rand1-1)%N,rand2] + spin[rand1,(rand2-1)%N]
                metroE = 2*detSpin*met

                if metroE < 0:
                    detSpin *= -1

                elif rand() < np.exp(-met*beta):
                    detSpin *= -1

                spin[rand1, rand2] = detSpin

    return spin

### Energy determines the energy at a determined spin configuration

def Energy(spin):

    energy = 0
    l = len(spin)

    for i in range(l):

        for j in range(l):

            updateSpin = spin[i,j]
            met = spin[(i+1)%N, j] + spin[i,(j+1)%N] + spin[(i-1)%N, j] + spin[i,(j-1)%N]
            energy += -met*updateSpin

    return energy/4.0

### Magnetization determines the magnetization at a determined spin configuration
def Magnetization(spin):

    mag = np.sum(spin)

    return mag

### simMetro utilizes the Metropolis algorithm to determine energy configuration at the simulation temperature

def simMetro(deltaE, simTemp):

  return np.exp((deltaE) / simTemp)

### animate updates spin configuration based on Monte Carlo to simulate magnetization

def animate(spin, simTemp):

  spin1 = np.copy(spin)
  randA = np.random.randint(spin.shape[0])
  randB = np.random.randint(spin.shape[1])
  spin1[randA, randB] *= -1
  energyNow = Energy(spin)
  energyUpdate = Energy(spin1)
  deltaE = energyNow - energyUpdate

  if  simMetro(deltaE, simTemp) > np.random.random():

    return spin1

  else:

    return spin

### mainPlot uses Energy and Magnetization functions to plot as a function of temperature
# def mainPlot():
#     for m in range(len(T)):
#         E1 = M1 = 0
#
#         for i in range(MCstep):
#             MCmetro(spin, 1.0/T[m])
#
#         for i in range(MCstep):
#             MCmetro(spin, 1.0/T[m])
#             Ene = Energy(spin)
#             Mag = Magnetization(spin)
#
#             E1 = E1 + Ene
#             M1 = M1 + Mag
#
#             MCEnergy[m]         = E1/(N**2)
#             MCMagnetization[m]  = M1/(N**2)
#
#     plt.plot(T, MCEnergy, 'g',label = 'green=energy')
#     plt.plot(T, abs(MCMagnetization), 'r',label = 'red=magnetization')
#     plt.legend()
#     plt.xlabel("Temperature (T)", fontsize=20)
#     plt.ylabel("Energy, Magnetization ", fontsize=20)
#     plt.show()
#
# mainPlot()

## Animation code ###
plt.show()

im = plt.imshow(spin, cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
t = 0
while True:
  if t % 10 == 0:
    im.set_data(spin)
    plt.draw()
  spin = animate(spin, simTemp)
  plt.pause(1e-10)
  t += 1

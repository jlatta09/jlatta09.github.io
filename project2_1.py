#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np
import time

A = 0.34839 # cross sectional area of wall element in m^2
rho = 2000.0 # density of wall material in kg / m^3
k = 0.6 # thermal conductivity of wall material in W / (m*K)
c = 900.0 # specific heat capacity in J / (kg*K)
sigma = 5.6704E-08 # Stefan-Boltzmann constant in W * m^-2 * K^-4
h = 30.0 # convective heat transfer coefficient in W / (m^2 * K)
l=0.1143
powDen = 0.54 # Power output in W/g of Pu
massPu = 4500
eps=0.11

T_init=500.0
T_inf=4.0
qin=100
total_time = 3600.0 # total duration of simulation in seconds


def simp_heat(Nx,Nt):
	dt = 10*total_time/Nt #ration of timestep in seconds
	dx=l/Nx
	x = np.linspace(0, dx*(Nx-1), Nx)
	y = np.linspace(0, dx*(Nx-1), Nx)
	timesamps = np.linspace(0, dt*Nt, Nt+1)
	heatfac = dx / (k*A)
	simfac = (k*dt) / (c*rho*dx*dx)
	Theat=np.zeros((20,20,10001))
	time1=np.linspace(0,total_time,Nt+1)

	for aa in range(len(x)):
		for bb in range(len(y)):
			for cc in range(len(timesamps)-1):
				Theat[10,aa,cc]=T_init
				Theat[bb,aa,0]=T_init#*np.exp(-x[aa])# I didn't find it realistic to set the middle of the array to 0 so I gave it a nonzero starting value.
				Theat[19,aa,0]=T_inf
				Theat[bb,0,0]=T_inf
				Theat[bb,19,0]=T_inf
	Q_dot_in = massPu*powDen
	fig=plt.figure()
	ax=fig.add_subplot(111)
	im=ax.imshow(Theat[:,:,0],cmap='hot',interpolation='nearest')
	fig.colorbar(im,shrink=0.5,aspect=5)
	plt.show(block=False)
	for j in range(len(timesamps)-1):
		for i in range(len(x)-2):
			for qq in range(len(y)-2):
				Theat[10,:,j]=T_init
				T_out = Theat[len(x)-1,0,j]
				Q_dot_out =eps* sigma * A * (T_out**4 - T_inf**4) + h * A * (T_out - T_inf)
				Theat[len(x)-1,len(y)-1, j+1] = T_out + simfac * (Theat[len(x)-2,len(y)-2, j] - T_out - heatfac * Q_dot_out)
				Theat[0,0, j+1] = Theat[0,0,j] + simfac * (Theat[1,1,j] - Theat[0,0,j] + heatfac * Q_dot_in-heatfac*Q_dot_out)
				Theat[i+1,qq+1,j+1] = Theat[i+1,qq+1,j] + simfac * (Theat[i,qq,j] - 2*Theat[i+1,qq+1,j] + Theat[i+2,qq+2,j]-heatfac*Q_dot_out)
				Theat[0,0,j]=0
				Theat[10,0,j]=0
				Theat[10,19,j]=0
		time.sleep(0.001)
		im.set_array(Theat[:,:,j])
		fig.canvas.draw()
	def static():
		one_arr=Theat[:,:,5000]
		print(one_arr)
		plt.imshow(one_arr,cmap='hot',interpolation='nearest')
		plt.show()
#	static()
plt.show()
simp_heat(20,10000)

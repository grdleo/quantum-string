import simulation
from edge import MirrorEdge, VoidEdge, ExcitatorSin, ExcitatorSinPeriod
import numpy as np
import os

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output" # A CHANGER BIEN SUR

duration = 5.0 # duration of simulation [s]
L = 1.0 # [m]
T = 10.0 # [N]
rho = 0.35 # [kg/m]
mass_particle = 0.05 # [kg]
pulsation_particle = 2*np.pi # [rad/s]
signal_frec = 10.0 # [Hz]
sampling_number = 50 # number of samples we will have in a single period (be careful with Shannon condition)

c = np.sqrt(T/rho)
signal_wavelen = c/signal_frec # [m]
signal_pulsation = 2*np.pi*signal_frec # [rad/s]

dt = 1/(sampling_number*signal_frec) # [s] computing the Δt according to Shannon condition
time_steps = int(duration/dt)
duration = dt*time_steps # recompute the duration in order to be consistent (may be a bit different)
dx = c*dt # [m] because of Δx/Δt=c
space_steps = int(L/dx)
L = dx*space_steps # recompute the length in order to be consistent (may be a bit different)

left = ExcitatorSinPeriod(dt, 0.01, signal_frec*2*np.pi, 0.0)
right = MirrorEdge()

simu = simulation.CenterFixed(dt, time_steps, space_steps, L, rho, T, left, right, mass_particle, pulsation_particle, log=True)
simu.run(mypath, anim=True, file=False, log=False)



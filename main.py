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
particle_freq = 1.0 # [Hz]
signal_freq = 5.0 # [Hz]
sampling_number = 100 # number of samples we will have in a single period (be careful with Shannon condition)

c = np.sqrt(T/rho)
signal_wavelen = c/signal_freq # [m]
pulsation_particle = 2*np.pi*particle_freq # [rad/s]
signal_pulsation = 2*np.pi*signal_freq # [rad/s]

relevant_freq = max(signal_freq, particle_freq) # the maximum relevant frequency in our problem
dt = 1/(sampling_number*relevant_freq) # [s] computing the Δt according to Shannon condition
time_steps = int(duration/dt)
duration = dt*time_steps # recompute the duration in order to be consistent (may be a bit different)
dx = c*dt # [m] because of Δx/Δt=c
space_steps = int(L/dx)
L = dx*space_steps # recompute the length in order to be consistent (may be a bit different)

left = ExcitatorSin(dt, 0.01, signal_pulsation, 0.0)
right = MirrorEdge()

### simu = simulation.FreeString(dt, time_steps, space_steps, L, rho, T, left, right, log=True)
simu = simulation.CenterFixed(dt, time_steps, space_steps, L, rho, T, left, right, mass_particle, pulsation_particle, log=True)
print(simu) # you can check if the simulation is good for you by printing it BEFORE running it...

simu.run(mypath, anim=True, file=False, frameskip=True, yscale=5.0, window_anim=False)



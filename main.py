import simulation
from edge import MirrorEdge, VoidEdge, ExcitatorSin, ExcitatorSinPeriod
import numpy as np
import os

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output" # A CHANGER BIEN SUR

duration = 5.0 # duration of simulation [s]
length = 1.0 # [m]
tension = 5.0 # [N]
density = 0.01 # [kg/m]
mass_particle = 0.01 # [kg]
particle_freq = 0.5*np.sqrt(25.0/mass_particle)/np.pi # [Hz]
signal_freq = 10.0 # [Hz]
sampling_number = 100 # number of samples we will have in a single period (be careful with Shannon condition)

celerity = np.sqrt(tension/density)
signal_wavelen = celerity/signal_freq # [m]
pulsation_particle = 2*np.pi*particle_freq # [rad/s]
signal_pulsation = 2*np.pi*signal_freq # [rad/s]

relevant_freq = max(signal_freq, particle_freq) # the maximum relevant frequency in our problem
dt = 1/(sampling_number*relevant_freq) # [s] computing the Δt according to Shannon condition
time_steps = int(duration/dt)
duration = dt*time_steps # recompute the duration in order to be consistent (may be a bit different)
dx = celerity*dt # [m] because of Δx/Δt=celerity
space_steps = int(length/dx)
length = dx*space_steps # recompute the length in order to be consistent (may be a bit different)

left = ExcitatorSin(dt, 0.01, signal_pulsation, 0.0)
right = MirrorEdge()

simu = simulation.FreeString(dt, time_steps, space_steps, length, density, tension, left, right, log=True)
### simu = simulation.CenterFixed(dt, time_steps, space_steps, length, density, tension, left, right, mass_particle, pulsation_particle, log=True)
# print(simu) # you can check if the simulation is good for you by printing it BEFORE running it...

simu.run(mypath, anim=True, file=True, frameskip=True, yscale=5.0, window_anim=False)
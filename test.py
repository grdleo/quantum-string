import simulation
import numpy as np
import os

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output" # A CHANGER BIEN SUR

dt = 0.01 # [s]
time_steps = 2500
L = 1.5 # [m]
T = 1 # [kg.m/s²]
rho = 1 # [kg/m]
string_steps = simulation.Simulation.compute_string_discret(L, T, rho, dt)
mass_particle = 0.05 # [kg]
pulsation_particle = 2*np.pi # [rad/s]

n_period_pulse = int(0.25*string_steps)

# excitations disponibles pour le bout de la corde
sinu = lambda tstep: 0.02*np.sin(2*np.pi*tstep/n_period_pulse)

simu = simulation.CenterFixed(dt, time_steps, L, rho, T, sinu, mass_particle, pulsation_particle, log=True)
simu.run(mypath, anim=True, file=False, log=True)


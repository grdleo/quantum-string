import simulation
import numpy as np
import os

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\leog\\Documents\\UGA\\M1\\Stage\\output" # A CHANGER BIEN SUR

dt = 0.001 # [s]
time_steps = 5000
L = 2.0 # [m]
c = 2.0 # [m/s]
string_steps = simulation.Simulation.compute_string_discret(L, c, dt)
string_density = 0.01 # [kg/m]
mass_particle = 0.005 # [kg]

n_period_pulse = int(0.25*string_steps)

# excitations disponibles pour le bout de la corde
sinu = lambda tstep: 0.02*np.sin(2*np.pi*tstep/n_period_pulse)

simu = simulation.CenterFixed(dt, time_steps, L, c, string_density, sinu, mass_particle, log=True)
simu.run(mypath, anim=True, file=True, log=True)
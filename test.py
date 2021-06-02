import simulation
import numpy as np
import os

curpath = os.path.dirname(os.path.abspath(__file__))

dt = 0.01 # [s]
time_steps = 30
L = 2.0 # [m]
c = 2.0 # [m/s]
string_steps = simulation.Simulation.compute_string_discret(L, c, dt)
string_density = 0.01 # [kg/m]
mass_particle = 0.005 # [kg]

n_period_pulse = int(0.25*string_steps)

# excitations disponibles pour le bout de la corde
pulse = lambda tstep: 0.2 if 0 <= tstep < 5 else 0
onesinu = lambda tstep: 0.2*np.sin(2*np.pi*tstep/n_period_pulse) if tstep < n_period_pulse else 0
sinu = lambda tstep: 0.2*np.sin(2*np.pi*tstep/n_period_pulse)

"""
On va ici utiliser la classe CenterFixedSimulation, qui permet de créer simplement une simulation
d'une corde avec une seule masselote en son centre.
Un .gif de visualisation de la simulation va être créé dans le répertoire où se trouve ce fichier
"""

simu = simulation.CenterFixed(dt, time_steps, L, c, string_density, sinu, mass_particle, log=True)
simu.make_anim(curpath, dpi=75)
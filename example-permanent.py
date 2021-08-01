from datetime import time
import os

from quantumstring.simulation import CenterFixed, FreeString, RingString, Simulation, Cavity
from quantumstring.edge import ExcitatorSinAbsorber, MirrorEdge, ExcitatorSin, AbsorberEdge
from quantumstring.particle import Particles, Particle
from quantumstring.process import PostProcess

import numpy as np
import matplotlib.pyplot as plt

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\grdle\\Documents\\UGA\\M1\\Stage\\output"

duration = 1.0 # [s]
space_steps = 511
length = 1.0 # [m]
tension = 1.0 # [N]
density = 0.005 # [kg/m]

c = np.sqrt(tension/density)
dx = length/space_steps
dt = dx/c
time_steps = int(duration/dt)
duration = dt*time_steps

examp = 0.05 # [m]
expuls = 2*np.pi*50 # [rad/s]
left = ExcitatorSinAbsorber(dt, examp, expuls)
right = AbsorberEdge()
ic0 = [0.0]*space_steps
ic1 = [0.0]*space_steps

pmass = 0.01 # [kg]
pk = 0.9*pmass*expuls**2
p1 = Particle(int(space_steps*0.5), 0.0, pmass, pk, True, space_steps)
ps = Particles(p1, space_steps=space_steps)

simu = Simulation(dt, time_steps, space_steps, length, density, tension, left, right, ic0, ic1, ps)

fpath, ppath, epath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r"),
    open(epath, "r")
)

process.anim(mypath)

fig, ax = plt.subplots(1, 2)
process.plot_particles(ax=ax[0], show=False, label=str(p1))
process.phasegraph_particles(ax=ax[1], show=False, label=str(p1))
plt.show()
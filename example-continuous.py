from datetime import time
import os

from quantumstring.simulation import CenterFixed, FreeString, RingString, Simulation, Cavity
from quantumstring.edge import ExcitatorSinAbsorber, MirrorEdge, ExcitatorSin, AbsorberEdge
from quantumstring.particle import Particles, Particle
from quantumstring.process import PostProcess

import numpy as np
import matplotlib.pyplot as plt

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output"

duration = 0.5 # [s]
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

xsteps = np.array([i for i in range(100, 500, 5)])
mass_profile = lambda xstep: 0.001*xstep**0 # constant profile
stiff_profile = lambda xstep: 1.0*xstep

particles_list = []

for m, k, x in zip(mass_profile(xsteps), stiff_profile(xsteps), xsteps):
    particles_list.append(
        Particle(x, 0.0, m, k, True, space_steps)
    )

ps = Particles(*particles_list, space_steps=space_steps)

simu = Simulation(dt, time_steps, space_steps, length, density, tension, left, right, ic0, ic1, ps)

fpath, ppath, epath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r"),
    open(epath, "r")
)

process.anim(mypath, frameskip=1)

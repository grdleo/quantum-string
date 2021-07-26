from datetime import time
import os

from quantumstring.simulation import Cavity, PulseRingString, RingString, Simulation
from quantumstring.edge import ExcitatorSinAbsorber, LoopEdge, MirrorEdge, ExcitatorSin, AbsorberEdge, ExcitatorWhiteNoise
from quantumstring.particle import Particles, Particle
from quantumstring.process import PostProcess

import numpy as np

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\leog\\Desktop\\lg2021stage\\output"

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

field_amp = 0.05 # [m]
left = ExcitatorWhiteNoise(dt, -0.05, 0.05, 0.2)
right = LoopEdge()

pmass = 0.01 # [kg]
pstiff = 100.0 # [N/m]
p = [
        Particle(int(space_steps*0.7), 0.0, pmass, pstiff, True, space_steps), # transparent particle
        Particle(int(space_steps*0.9), 0.0, pmass, pstiff, True, space_steps, color=(255, 0, 0)) # non transparent particle
]
ps = Particles(*p, space_steps=space_steps)

rest_string = [0.0]*space_steps
simu = Simulation(dt, time_steps, space_steps, length, density, tension, left, right, rest_string.copy(), rest_string.copy(), ps)
fpath, ppath, epath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r"),
    open(epath, "r")
)

process.anim(mypath, frameskip=1)
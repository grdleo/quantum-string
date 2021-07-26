from datetime import time
import os

from quantumstring.simulation import Cavity, PulseRingString
from quantumstring.edge import ExcitatorSinAbsorber, MirrorEdge, ExcitatorSin, AbsorberEdge
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
field_wavelen_factor = 0.5

field_wavelen = length*field_wavelen_factor
field_k = 2*np.pi/field_wavelen
field_omega = c*field_k
field_generator = lambda x, t: field_amp*np.sin(field_k*x)*np.cos(field_omega*t)

xlin = np.linspace(0.0, length, space_steps)
ic0 = field_generator(xlin, 0.0)
ic1 = field_generator(xlin, dt)

pmass = 0.01 # [kg]
p = [
        Particle(int(space_steps*0.7), 0.0, pmass, pmass*field_omega**2, True, space_steps), # transparent particle
        Particle(int(space_steps*0.9), 0.0, pmass, 100.0, True, space_steps, color=(255, 0, 0)) # non transparent particle
]
ps = Particles(*p, space_steps=space_steps)

simu = Cavity(dt, time_steps, space_steps, length, density, tension, ic0, ic1, ps)
fpath, ppath, epath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r"),
    open(epath, "r")
)

process.anim(mypath, frameskip=1)
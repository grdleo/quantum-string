from datetime import time
import os

from quantumstring.simulation import PulseRingString
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

gauss_amplitude = 0.05 # [m]
gauss_bandwidth = 0.03 # [m]
gauss_wavelength = 0.01 # [m]
gauss_pulsation = 2*np.pi*c/gauss_wavelength # [rad/s]

pmass = 0.001 # [kg]
p = [
        Particle(int(space_steps*0.7), 0.0, pmass, pmass*gauss_pulsation**2, True, space_steps), # transparent particle
        Particle(int(space_steps*0.9), 0.0, pmass, 100.0, True, space_steps, color=(255, 0, 0)) # non transparent particle
]
ps = Particles(*p, space_steps=space_steps)

simu = PulseRingString(dt, time_steps, space_steps, length, density, tension, gauss_amplitude, gauss_bandwidth, gauss_wavelength, ps)
fpath, ppath, epath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r"),
    open(epath, "r")
)

process.anim(mypath, frameskip=10)
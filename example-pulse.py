from datetime import time
import os

from quantumstring.simulation import Simulation
from quantumstring.edge import ExcitatorSinAbsorber, LoopEdge, MirrorEdge, ExcitatorSin, AbsorberEdge
from quantumstring.particle import Particles, Particle
from quantumstring.process import PostProcess

import numpy as np
from scipy import signal 

mypath = os.path.dirname(os.path.abspath(__file__))
mypath = "C:\\Users\\grdle\\Documents\\UGA\\M1\\Stage\\output"

duration = 2.5 # [s]
space_steps = 511
length = 1.0 # [m]
tension = 1.0 # [N]
density = 0.005 # [kg/m]

c = np.sqrt(tension/density)
dx = length/space_steps
dt = dx/c
time_steps = int(duration/dt)
duration = dt*time_steps

gauss_amplitude = 0.15 # [m]
gauss_bandwidth = 0.05 # [m]
gauss_wavelength = 0.03 # [m]
gauss_pulsation = 2*np.pi*c/gauss_wavelength # [rad/s]

x = np.linspace(0.0, length, space_steps) - 5*gauss_bandwidth
gauss = gauss_amplitude*np.exp(-0.5*(x/gauss_bandwidth)**2)
sine = np.sin(2*np.pi/gauss_wavelength*x)
ic0 = list(gauss_amplitude*gauss*sine)
ic1 = ic0.copy()
ic1.insert(0, 0.0)
ic1.pop(-1)

pmass = 0.001 # [kg]
p = [
        Particle(int(space_steps*0.7), 0.0, pmass, pmass*gauss_pulsation**2, True, space_steps), # transparent particle
        Particle(int(space_steps*0.9), 0.0, pmass, 100.0, True, space_steps, color=(255, 0, 0)) # non transparent particle
]
ps = Particles(*p, space_steps=space_steps)

left = LoopEdge()
right = LoopEdge()

simu = Simulation(dt, time_steps, space_steps, length, density, tension, left, right, ic0, ic1, ps)
fpath, ppath, epath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r"),
    open(epath, "r")
)

process.anim(mypath, yscale=5.0, frameskip=10)
process.energy()
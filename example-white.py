from datetime import time
import os

from quantumstring.simulation import Cavity, PulseRingString, RingString, Simulation
from quantumstring.edge import ExcitatorSinAbsorber, LoopEdge, MirrorEdge, ExcitatorSin, AbsorberEdge, ExcitatorWhiteNoise
from quantumstring.particle import Particles, Particle
from quantumstring.process import PostProcess
from quantumstring.phystring import PhyString

import numpy as np

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

left = MirrorEdge()
right = MirrorEdge()

xline = np.linspace(0.0, length, space_steps)

def white_excitator(x: list, freqs: list):
    phases = 2*np.pi*np.random.rand(x.size)
    sines = np.array([np.sin(f*x + p) for f, p in zip(freqs, phases)])
    r = np.mean(sines, axis=0)
    if r.shape != x.shape:
        raise AssertionError("Wrong axis... Léo should correct this")
    return r

rel2xstep = lambda rel: int(space_steps*rel) # returns the cell index corresponding to the relative position

pmass = 0.01 # [kg]
pstiff = 100.0 # [N/m]
p = [
        Particle(rel2xstep(0.7), 0.0, pmass, pstiff, True, space_steps), # transparent particle
        Particle(rel2xstep(0.9), 0.0, pmass, pstiff, True, space_steps, color=(255, 0, 0)) # non transparent particle
]
ps = Particles(*p, space_steps=space_steps)

nb_white = 100
white_freqs = np.array([i for i in range(0, nb_white)]).astype(float)
a, b = 0, rel2xstep(0.6)
ic0 = xline.copy()
ic0[a:b] = white_excitator(xline[a:b], white_freqs)*0.05
ic1 = PhyString.shift_list_right(ic0)

print(ic1)

simu = Simulation(dt, time_steps, space_steps, length, density, tension, left, right, ic0, ic1, ps)
fpath, ppath, epath = simu.run(mypath)

process = PostProcess(
    open(fpath, "r"),
    open(ppath, "r"),
    open(epath, "r")
)

process.anim(mypath, frameskip=1)
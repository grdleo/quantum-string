import numpy as np
import random

class Edge:
    """
        A class to handle edges conditions of the string
    """

    def __init__(self, dt: float):
        self.dt = dt

class VoidEdge(Edge):
    def __init__(self, dt: float):
        super().__init__()
        self.excitation = lambda tstep: None

class MirrorEdge(Edge):
    def __init__(self, dt: float):
        super().__init__()
        self.excitation = lambda tstep: 0.0

class ExcitatorEdge(Edge):
    def __init__(self, dt: float, excitation):
        super().__init__()
        self.excitation = excitation

class ExcitatorSin(ExcitatorEdge):
    def __init__(self, dt: float, amplitude: float, pulsation: float, phase: float):
        sin = lambda tstep: amplitude*np.sin(pulsation*tstep*dt + phase)
        super().__init__(dt, sin)

class ExcitatorSinPeriod(ExcitatorEdge):
    def __init__(self, dt: float, amplitude: float, pulsation: float, phase: float, nb_periods: int):
        sin = lambda tstep: amplitude*np.sin(pulsation*tstep*dt + phase) if tstep*dt <= nb_periods*2*np.pi/pulsation else 0.0
        super().__init__(dt, sin)

class ExcitatorPulse(ExcitatorEdge):
    def __init__(self, dt: float, amplitude: float, duration: float):
        pulse = lambda tstep: amplitude if tstep*dt <= duration else 0.0 
        super().__init__(dt, pulse)

class ExcitatorWhiteNoise(ExcitatorEdge):
    def __init__(self, dt: float, amp_min: float, amp_max: float, duration: float):
        amp_delta = np.abs(amp_max - amp_min)
        noise = lambda tstep: random.random()*amp_delta + amp_min if tstep*dt <= duration else 0.0
        super().__init__(dt, noise)

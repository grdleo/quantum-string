import numpy as np
import random

class Edge:
    """
        A class to handle edges conditions of the string
    """

    def __init__(self):
        self.condition = None

class VoidEdge(Edge):
    def __init__(self):
        super().__init__()
        self.condition = lambda tstep: None

class MirrorEdge(Edge):
    def __init__(self):
        super().__init__()
        self.condition = lambda tstep: 0.0

class ExcitatorEdge(Edge):
    def __init__(self, excitation):
        super().__init__()
        self.condition = excitation
    
    def __add__(self, other):
        sumcond = lambda tstep: self.condition(tstep) + other.condition(tstep)
        return ExcitatorEdge(sumcond)
    
    def __sub__(self, other):
        difcond = lambda tstep: self.condition(tstep) - other.condition(tstep)
        return ExcitatorEdge(difcond)
    
    def __mul__(self, other):
        prodcond = lambda tstep: self.condition(tstep)*other.condition(tstep)
        return ExcitatorEdge(prodcond)

class ExcitatorSin(ExcitatorEdge):
    def __init__(self, dt: float, amplitude: float, pulsation: float, delay: float):
        steps_delay = int(np.round(delay/dt))
        def sin(tstep):
            delayed = tstep - steps_delay
            return amplitude*np.sin(pulsation*delayed*dt) if delayed >= 0 else 0.0
        super().__init__(sin)

class ExcitatorSinPeriod(ExcitatorEdge):
    def __init__(self, dt: float, amplitude: float, pulsation: float, delay: float, nb_periods=1):
        steps_delay = int(np.round(delay/dt))
        def sin(tstep):
            delayed = tstep - steps_delay
            return amplitude*np.sin(pulsation*delayed*dt) if delayed >= 0 and delayed*dt <= nb_periods*2*np.pi/pulsation else 0.0
        super().__init__(sin)

class ExcitatorPulse(ExcitatorEdge):
    def __init__(self, dt: float, amplitude: float, duration: float):
        pulse = lambda tstep: amplitude if tstep*dt <= duration else 0.0 
        super().__init__(pulse)

class ExcitatorWhiteNoise(ExcitatorEdge):
    def __init__(self, dt: float, amp_min: float, amp_max: float, duration: float):
        amp_delta = np.abs(amp_max - amp_min)
        noise = lambda tstep: random.random()*amp_delta + amp_min if tstep*dt <= duration else 0.0
        super().__init__(noise)

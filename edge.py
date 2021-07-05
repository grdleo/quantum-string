import numpy as np

import random
from typing import Callable

"""
    Classes for dealing with the end of the string
"""

class Edge:
    """
        A class to handle edges conditions of the string
    """

    def __init__(self, condition: Callable):
        self.condition = condition

    def __repr__(self):
        return "Edge not specified"

class MirrorEdge(Edge):
    """
        Inheritence from Edge: simulate a perfect mirror (field equal to zero at that point)
    """
    def __init__(self):
        condition = lambda tstep: 0.0
        super().__init__(condition)
    
    def __repr__(self):
        return "Edge: mirror"

class AbsorberEdge(Edge):
    """
        Inheritence from Edge: simulate an absorber
    """
    def __init__(self):
        super().__init__(None)
    
    def __repr__(self):
        return "Edge: absorber"

class LoopEdge(Edge):
    """
        Inheritence from Edge: make the string a loop, by making the two edges one
        considering my code, this is equivalent with no condition at all
    """
    def __init__(self):
        super().__init__(None)
    
    def __repr__(self):
        return "Edge: loop"

class ExcitatorEdge(Edge):
    pass

class ExcitatorEdge(Edge):
    """
        Inheritence from Edge: base class for an excitator, gives definition for operators 
    """

    infostring: str
    """ Information about the excitator """

    def __init__(self, excitation):
        self.infostring = "Excitator: not yet defined"
        super().__init__(excitation)
    
    def __add__(self, other: ExcitatorEdge) -> ExcitatorEdge:
        sumcond = lambda tstep: self.condition(tstep) + other.condition(tstep)
        infostring = "({}+{})".format(self.infostring, other.infostring)
        ex = ExcitatorEdge(sumcond)
        ex.infostring = infostring
        return ex
    
    def __sub__(self, other: ExcitatorEdge) -> ExcitatorEdge:
        difcond = lambda tstep: self.condition(tstep) - other.condition(tstep)
        infostring = "({}-{})".format(self.infostring, other.infostring)
        ex = ExcitatorEdge(difcond)
        ex.infostring = infostring
        return ex
    
    def __mul__(self, other: ExcitatorEdge) -> ExcitatorEdge:
        prodcond = lambda tstep: self.condition(tstep)*other.condition(tstep)
        infostring = "({}×{})".format(self.infostring, other.infostring)
        ex = ExcitatorEdge(prodcond)
        ex.infostring = infostring
        return ex
    
    def __repr__(self):
        return self.infostring

class ExcitatorSin(ExcitatorEdge):
    def __init__(self, dt: float, amplitude: float, pulsation: float, delay: float):
        """
            Creates a sinusoidal excitator

            :param dt: Δt of the simulation [s]
            :param amplitude: amplitude of the sine [m]
            :param pulsation: pulsation of the sine [rad/s]
            :param delay: time to wait before starting the excitator [s]
        """
        steps_delay = int(np.round(delay/dt))
        def sin(tstep):
            delayed = tstep - steps_delay
            return amplitude*np.sin(pulsation*delayed*dt) if delayed >= 0 else 0.0
        self.infostring = "sin[amplitude={}, ω={}{}]".format(amplitude, pulsation, "" if delay == 0.0 else ", delay={}".format(delay))
        super().__init__(sin)

class ExcitatorSinPeriod(ExcitatorEdge):
    def __init__(self, dt: float, amplitude: float, pulsation: float, delay: float, nb_periods=1):
        """
            Creates a sinusoidal excitator, but excite inly for a given number of periods

            :param dt: Δt of the simulation [s]
            :param amplitude: amplitude of the sine [m]
            :param pulsation: pulsation of the sine [rad/s]
            :param delay: time to wait before starting the excitator [s]
            :param nb_periods: number of full periods to excite
        """
        steps_delay = int(np.round(delay/dt))
        def sin(tstep):
            delayed = tstep - steps_delay
            return amplitude*np.sin(pulsation*delayed*dt) if delayed >= 0 and delayed*dt <= nb_periods*2*np.pi/pulsation else 0.0
        self.infostring = "sin[periods={}, amplitude={}, ω={}{}]".format(nb_periods, amplitude, pulsation, "" if delay == 0.0 else ", delay={}".format(delay))
        super().__init__(sin)

class ExcitatorPulse(ExcitatorEdge):
    """
        Creates a pulse excitator with a given amplitude and duration

        :param dt: Δt of the simulation [s]
        :param amplitude: amplitude of the pulse [m]
        :param duration: duration of the pulse [s]
    """
    def __init__(self, dt: float, amplitude: float, duration: float):
        pulse = lambda tstep: amplitude if tstep*dt <= duration else 0.0 
        self.infostring = "pulse[amplitude={}, T={}]".format(amplitude, duration)
        super().__init__(pulse)

class ExcitatorWhiteNoise(ExcitatorEdge):
    def __init__(self, dt: float, amp_min: float, amp_max: float, duration: float):
        amp_delta = np.abs(amp_max - amp_min)
        noise = lambda tstep: random.random()*amp_delta + amp_min if tstep*dt <= duration else 0.0
        self.infostring = "white[min={}, max={}".format(amp_min, amp_max)
        super().__init__(noise)

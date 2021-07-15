from os import abort
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

    def __init__(self, condition: Callable, absorber=False, loop=False):
        self.condition = condition
        self.absorber = absorber
        self.loop = loop

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
        return "mirror"

class AbsorberEdge(Edge):
    """
        Inheritence from Edge: simulate an absorber
    """
    def __init__(self):
        condition = lambda tstep: 0.0
        super().__init__(condition, absorber=True)
    
    def __repr__(self):
        return "absorber"

class LoopEdge(Edge):
    """
        Inheritence from Edge: make the string a loop, by making the two edges one
        considering my code, this is equivalent with no condition at all
    """
    def __init__(self):
        super().__init__(None, loop=True)
    
    def __repr__(self):
        return "loop"

class ExcitatorEdge(Edge):
    pass

class ExcitatorEdge(Edge):
    """
        Inheritence from Edge: base class for an excitator, gives definition for operators 
    """

    infostring: str
    """ Information about the excitator """

    def __init__(self, excitation, absorber=False):
        self.infostring = "Excitator: not yet defined"
        super().__init__(excitation, absorber=absorber)
    
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
    def __init__(self, dt: float, amplitude: float, pulsation: float, delay: float, absorber=False):
        """
            Creates a sinusoidal excitator

            :param dt: Δt of the simulation [s]
            :param amplitude: amplitude of the sine [m]
            :param pulsation: pulsation of the sine [rad/s]
            :param delay: time to wait before starting the excitator [s]
        """
        steps_delay = int(np.round(delay/dt))
        def sin(tstep: int):
            delayed = tstep - steps_delay
            return amplitude*np.sin(pulsation*delayed*dt) if delayed >= 0 else 0.0
        super().__init__(sin, absorber=absorber)
        self.infostring = "sin[A={:.3f}m, ω={:.1f}rad/s{}]".format(amplitude, pulsation, "" if delay == 0.0 else ", delay={}".format(delay))

class ExcitatorSinAbsorber(ExcitatorSin):
    def __init__(self, dt: float, amplitude: float, pulsation: float):
        """
            !!! because of calculations for absorbing, the 'effective' sine that is propagating has his amplitude HALVED...
            !!! so that what user is entering is consistent, we double the amplitude the user is entering...
        """
        super().__init__(dt, 2.0*amplitude, pulsation, 0.0, absorber=True)
        self.infostring = "sin[A={:.3f}m, ω={:.1f}rad/s] & absorber".format(amplitude, pulsation)

class ExcitatorPulse(ExcitatorEdge):
    """
        Creates a pulse excitator with a given amplitude and duration

        :param dt: Δt of the simulation [s]
        :param amplitude: amplitude of the pulse [m]
        :param duration: duration of the pulse [s]
    """
    def __init__(self, dt: float, amplitude: float, duration: float, absorber=False):
        pulse = lambda tstep: amplitude if tstep*dt <= duration else 0.0 
        super().__init__(pulse, absorber=absorber)
        self.infostring = "pulse[A={:.3f}m, T={:.1f}s]".format(amplitude, duration)

class ExcitatorWhiteNoise(ExcitatorEdge):
    def __init__(self, dt: float, amp_min: float, amp_max: float, duration: float, absorber=False):
        amp_delta = np.abs(amp_max - amp_min)
        noise = lambda tstep: random.random()*amp_delta + amp_min if tstep*dt <= duration else 0.0
        super().__init__(noise, absorber=absorber)
        self.infostring = "white[{:.3f}m, {:.3f}m".format(amp_min, amp_max)

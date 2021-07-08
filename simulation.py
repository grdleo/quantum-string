from __future__ import annotations
import os
import math
import datetime
import json

from phystring import PhyString
from particle import Particle, Particles
from edge import Edge, MirrorEdge, LoopEdge

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import ffmpeg

"""
    Classes for making simulations of the system
"""

class Simulation:
    """
        Class that wraps the whole simulation thing
    """
    IMG_PREFIX = "qs_img___"
    VID_PREFIX = "QuantumString"
    IMG_FORMAT = "png"
    PERCENT_MAX = 256

    STR_DT = "dt"
    STR_DX = "dx"
    STR_TIMESTEPS = "nt"
    STR_SPACESTEPS = "nx"
    STR_TENSION = "T"
    STR_DENSITY = "rho"
    STR_LENGTH = "L"
    STR_EDGE_LEFT = "edge_left"
    STR_EDGE_RIGHT = "edge_right"

    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (64, 64, 64)
    WHITE = (255, 255, 255)

    def __init__(self, dt: float, time_steps: int, space_steps: int, string_len: float, string_density: float, string_tension: float, edge_left: Edge, edge_right: Edge, ic0: list[float], ic1: list[float], particles, memory_field=5, log=True):
        """
            Initialisation of the simulation

            :param dt: value of the time step [s]
            :param time_steps: number of time steps
            :param space_steps: number of cells in the string
            :param string_len: length of the string [m]
            :param string_c: celerity of the string [m/s]
            :param string_density: linear density of the string [kg/m]
            :param edge_left: the condition at the left extremity of the string
            :param edge_right: the condition at the right extremity of the string
            :param ic_pos: initial condition of the position of the string
            :param ic_vel: initial condition of the velocity of the string
            :param particles: Particles object
            :param memory_field:
            :param log: if True, prints the simulation loading
        """
        self.log = log
        self.time_steps = time_steps
        self.dt = dt
        self.time = str(datetime.datetime.now())

        self.s = PhyString(string_len, space_steps, dt, string_density, string_tension, edge_left, edge_right, ic0, ic1, particles, memory_field=memory_field)
    
    def infos(self) -> dict[str]:
        """
            Returns a dictionary containing the informations about the simulation
        """
        return {
            "desc": "QUANTUM STRING SIMULATION",
            "date": self.time,
            Simulation.STR_DT: self.s.dt,
            Simulation.STR_DX: self.s.dx,
            Simulation.STR_TIMESTEPS: self.time_steps,
            Simulation.STR_SPACESTEPS: self.s.space_steps,
            Simulation.STR_LENGTH: self.s.length,
            Simulation.STR_TENSION: self.s.tension,
            Simulation.STR_DENSITY: self.s.linear_density,
            Simulation.STR_EDGE_LEFT: str(self.s.edge_left),
            Simulation.STR_EDGE_RIGHT: str(self.s.edge_right),
            "particles": self.s.particles.infos()
        }
    
    def __repr__(self):
        return "[SIMULATION]    Δt={}s, Δx={}m, time steps={}, space steps={}\n{}\n{}\nEstimation of simulation size = {}MB".format(
            self.s.dt, 
            self.s.dx, 
            self.time_steps, 
            self.s.space_steps,
            self.s,
            self.s.particles,
            self.size_estimation_mb())

    def run(self, path: str) -> tuple[str, str]:
        """
            Runs the simulation with options to save it as a animation and/or in a file
            Returns the path of the field and particles file generated (if generated)

            :param path: location where to save the outputs
        """
        dtnow = datetime.datetime.now()
        timestamp = int(dtnow.timestamp())
        jsoninfos = json.dumps(self.infos())
        print(self) if self.log else None

        field_file_path = os.path.join(path, "QuantumString-field_{}.txt".format(timestamp))
        particles_file_path = os.path.join(path, "QuantumString-particles_{}.txt".format(timestamp))

        begtxt = "{}\n".format(jsoninfos)
        ff = open(field_file_path, "w", encoding="utf-8")
        pf = open(particles_file_path, "w", encoding="utf-8")
        ff.write(begtxt)
        pf.write(begtxt)

        ts = datetime.datetime.now()
        percent = 0
        list_dt_compute = []
        for t in range(0, self.time_steps):
            if t > 1: # do not update when the timesteps are lower than 1 bc this corresponds to the two initial fields
                self.s.update() # update the string
            
            # printing the update to the console...
            prop = t/self.time_steps
            newpercent = math.floor(prop*Simulation.PERCENT_MAX)
            if (newpercent != percent) and self.log: # update the console
                newts = datetime.datetime.now()
                dtcompute = (newts - ts).total_seconds()
                elapsed = sum(list_dt_compute)
                list_dt_compute.append(dtcompute)
                spinner = "←↖↑↗→↘↓↙" # ".ₒoO0Ooₒ" # "+÷–÷" # "+×" #
                load = percent % len(spinner)
                print("{:2}% {} {:.4f}s left                ".format(int(percent/Simulation.PERCENT_MAX*100), spinner[load:load+1],  float(elapsed*(1/prop-1))), end="\r")
                ts = newts
            percent = newpercent
            # ...
            
            # writing the fields in the files
            f = self.s.field.get_val_time(t)
            pp = self.s.particles.list_pos(tstep=t)
            fstr = Simulation.list2str(f)
            pstr = Simulation.list2str(pp)
            ff.write("{}\n".format(fstr))
            pf.write("{}\n".format(pstr))
            # ...
        print("")

        return field_file_path, particles_file_path
    
    def size_estimation_mb(self) -> float:
        """
            Gives an estimation of the final size of the field file, in MB (Mo in french)
        """
        bytes_per_cell = 4
        return bytes_per_cell*self.s.space_steps*self.time_steps*1e-6

    @staticmethod
    def list2str(l: list) -> str:
        """
            Converts a list to a string in a simple format for our use

            ex: 
            >>> list2str([1, 2, 3])
            '1,2,3'
        """
        return str(list(l)).replace("[", "").replace("]", "").replace(" ", "").replace("\n", "")
    
    @staticmethod
    def str2list(s: str, type=float) -> list:
        """
            Converts a converted string back into a NumPy array. The type of values can be specified

            ex:
            >>> str2list("1,2,3")
            [1, 2, 3]
        """
        s = s.replace("\n", "")
        l = s.split(",")
        return np.array(l).astype(type) if s != "" else np.array([])

#################################################################

class RestString(Simulation):
    """
        Abstraction of Simulation: the initial field is at rest
    """
    def __init__(self, dt: float, time_steps: int, space_steps: int, string_len: float, string_density: float, string_tension: float, edge_left: Edge, edge_right: Edge, particles, log=True, memory_field=5):
        ic0 = [0.0]*space_steps
        ic1 = ic0.copy()
        super().__init__(dt, time_steps,  space_steps, string_len, string_density, string_tension, edge_left, edge_right, ic0, ic1, particles, log=log, memory_field=memory_field)

class FreeString(RestString):
    """
        Abstraction of RestString: the system is particle free
    """
    def __init__(self, dt: float, time_steps: int, space_steps: int, string_len: float, string_density: float, string_tension: float, edge_left: Edge, edge_right: Edge, log=True, memory_field=5):
        particles = Particles(space_steps=space_steps)
        super().__init__(dt, time_steps, space_steps, string_len, string_density, string_tension, edge_left, edge_right, particles, log=log, memory_field=memory_field)

class CenterFixed(RestString):
    """
        Abstraction of RestString: the system has a single particle in the center of the string
    """
    def __init__(self, dt: float, time_steps: int, space_steps: int, string_len: float, string_density: float, string_tension: float, edge_left: Edge, edge_right: Edge, mass_particle: float, pulsation_particle: float, log=True, memory_field=5):
        center_string = math.floor(space_steps*0.5)
        p = Particle(center_string, 0.0, mass_particle, pulsation_particle, True, space_steps)
        particles = Particles(p)
        super().__init__(dt, time_steps, space_steps, string_len, string_density, string_tension, edge_left, edge_right, particles, log=log, memory_field=memory_field)

class Cavity(Simulation):
    """
        Abstraction of Simulation: mirrors in both ends, initial position given but initial velocity is zero
    """
    def __init__(self, dt: float, time_steps: int, space_steps: int, string_len: float, string_density: float, string_tension: float, ic0: list[float], ic1: list[float], particles: Particles, log=True, memory_field=5):
        ml, mr = MirrorEdge(), MirrorEdge()
        super().__init__(dt, time_steps, space_steps, string_len, string_density, string_tension, ml, mr, ic0, ic1, particles, log=log, memory_field=memory_field)

class RingString(Simulation):
    """
        Abstraction of Simulation: both ends are connected (equivalent of a ring)
    """
    def __init__(self, dt: float, time_steps: int, space_steps: int, string_len: float, string_density: float, string_tension: float, ic0: list[float], ic1: list[float], particles: Particles, log=True, memory_field=5):
        ll, lr = LoopEdge(), LoopEdge()
        super().__init__(dt, time_steps, space_steps, string_len, string_density, string_tension, ll, lr, ic0, ic1, particles, log=log, memory_field=memory_field)
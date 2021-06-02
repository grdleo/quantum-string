from phystring import PhyString
from particle import Particle, Particles

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import math
import time
import random
import time

class Simulation:
    """
        Class that wraps the whole simulation thing
    """
    def __init__(self, dt: float, time_steps: int, string_len: float, string_c: float, string_density: float, exitation_fx, ic_pos: list, ic_vel: list, particles, log=True):
        """
            Initialisation of the simulation

            :param dt: value of the time step [s]
            :param time_steps: number of time steps
            :param string_len: length of the string [m]
            :param string_c: celerity of the string [m/s]
            :param string_density: linear density of the string [kg/m]
            :param excitation_fx: function corresponding to the condition at the left side of the string
            :param ic_pos: initial condition of the position of the string
            :param ic_vel: initial condition of the velocity of the string
            :param particles: Particles object
            :param log: if True, prints the simulation loading

            :type dt: float
            :type time_steps: int
            :type string_len: float
            :type string_c: float
            :type string_density: float
            :type excitation_fx: function
            :type ic_pos: list
            :type ic_vel: list
            :type particles: Particles object
            :type log: bool

        """
        string_discret = Simulation.compute_string_discret(string_len, string_c, dt)
        self.log = log
        self.time_steps = time_steps
        self.dt = dt

        self.s = PhyString(string_len, string_discret, dt, string_c, string_density, exitation_fx, ic_pos, ic_vel, particles)

        print("SIMULATION:") if self.log else None
        for t in range(0, time_steps):
            self.s.update()
            print("{}/{}".format(t, time_steps)) if self.log else None
    
    @staticmethod
    def compute_string_discret(l: float, c: float, dt: float):
        """
            Returns the number of cells needed for the simulation, considering the length, the celerity and the time step
            Follows the equation Δx/Δt = c

            :param l: length of the string [m]
            :param c: celerity of the string [m/s]
            :param dt: time step [s]

            :return: number n of cells required for the string, such as l = n Δx
            :rtype: int
        """
        return int(l/c/dt)
    
    def print(self):
        """
            Prints a quick animation of the simulation in the console
        """
        clear_console = lambda: os.system("cls")
        for f in self.s.field.val:
            len_char = self.s.dx # [m]
            nb_char = self.s.nb_linear_steps
            interval = max(f) + 1e-6
            min_val = min(f)
            while interval >= min_val:
                upper = interval
                delta = 0.5*len_char
                mid = upper - delta
                vals_in_row = np.where(np.abs(f - mid) < delta)[0]
                row = [" "]*nb_char
                for i in vals_in_row:
                    row[i] = "."
                interval -= len_char
                print("".join(row), end="\n")
            time.sleep(self.s.dt)
            clear_console()
    
    def make_anim(self, path: str, dpi=96):
        """
            Creates an animation (using MatPlotLib) for the simulation, in a .gif 

            :param path: path of the folder for the file to be created
            :param dpi: dpi for the gif

            :type path: str
            :type dpi: int
        """
        dpi = 72 
        x = np.linspace(0, self.s.length, self.s.nb_linear_steps)
        file_prefix = "QUANTUMSTRING_anim"
        anim = []
        list_names = []
        print("ANIMATION CREATION:") if self.log else None
        for i in range(0, self.time_steps):
            fn = "{}\\{}-{}.png".format(path, file_prefix, i)
            f = self.s.field.get_val_time(i)
            p = self.s.particles.mass_presence(tstep=i)
            pp = x*p # == 0 if no particle, == to the x position if the particle is in the cell
            px = pp[pp != 0] # get a list of the x positions of the particle
            py = f[pp != 0] # get a list of the y positions of the particle
            
            plt.plot(x, f, "b")
            plt.plot(px, py, "r.")
            plt.title("Field visualisation   t={}s".format(i*self.s.dt))
            plt.xlabel("$x$ position on the bench [m]")
            plt.ylabel("$y$ value for the field [m]")
            plt.ylim((-0.6, 0.6))
            plt.savefig(fn, dpi=dpi)
            plt.close()
            anim.append(Image.open(fn))
            list_names.append(fn)
            print("{}/{}".format(i, self.time_steps)) if self.log else None
        anim[0].save("{}\\{}-{}.gif".format(path, file_prefix, int(time.time())), duration=self.dt, save_all=True, append_images=anim[1:], optimize=True, loop=0)
        for i, fn in zip(anim, list_names):
            i.close()
            os.remove(fn)

class RestString(Simulation):
    """
        Abstraction of Simulation: the initial field is at rest
    """
    def __init__(self, dt: float, time_steps: int, string_len: float, string_c: float, string_density: float, exitation_fx, particles, log=True):
        string_discret = Simulation.compute_string_discret(string_len, string_c, dt)
        ic_pos = [0]*string_discret
        ic_vel = ic_pos.copy()
        super().__init__(dt, time_steps, string_len, string_c, string_density, exitation_fx, ic_pos, ic_vel, particles, log=log)

class FreeString(RestString):
    """
        Abstraction of RestString: the system is particle free
    """
    def __init__(self, dt: float, time_steps: int, string_len: float, string_c: float, string_density: float, exitation_fx, log=True):
        string_discret = Simulation.compute_string_discret(string_len, string_c, dt)
        particles = Particles(string_discret, [])
        super().__init__(dt, time_steps, string_len, string_c, string_density, exitation_fx, particles, log=log)

class CenterFixed(RestString):
    """
        Abstraction of RestString: the system has a single particle in the center of the string
    """
    def __init__(self, dt: float, time_steps: int, string_len: float, string_c: float, string_density: float, exitation_fx, mass_particle: float, log=True):
        string_discret = Simulation.compute_string_discret(string_len, string_c, dt)
        center_string = math.floor(string_discret*0.5)
        p = Particle(center_string, 0.0, mass_particle, 1.0, True, string_discret)
        particles = Particles(string_discret, [p])
        super().__init__(dt, time_steps, string_len, string_c, string_density, exitation_fx, particles, log=log)
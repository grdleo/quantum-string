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
import datetime

class Simulation:
    """
        Class that wraps the whole simulation thing
    """
    IMG_PREFIX = "QuantumString"
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
    
    def __repr__(self):
        return "[SIMULATION]    Δt={}s, Δx={}m, time steps={}, string steps (nb discretisation)={}; ".format(self.s.dt, self.s.dx, self.time_steps, self.s.nb_linear_steps)
    
    def run(self, path: str, anim=True, file=True, log=True, dpi=96):
        """
            Runs the simulation with options to save it as a animation and/or in a file

            :param path: path for the simulation to be saved
            :type path: str
        """
        dtnow = datetime.datetime.now()
        timestamp = int(dtnow.timestamp())
        print("SIMULATION:") if self.log else None
        ffn = "QuantumString-field_{}.txt".format(timestamp)
        pfn = "QuantumString-particles_{}.txt".format(timestamp)
        idtxtfield = "QUANTUM STRING SIMULATION ({}): {} {}\n".format(dtnow.isoformat(), self, self.s)
        idtxtparts = "QUANTUM STRING SIMULATION ({}): {} {}\n".format(dtnow.isoformat(), self, self.s.particles)
        ff = open("{}\\{}".format(path, ffn), "w", encoding="utf-8")
        pf = open("{}\\{}".format(path, pfn), "w", encoding="utf-8")
        ff.write(idtxtfield)
        pf.write(idtxtparts)
        
        list_img = []
        names_img = []
        for t in range(0, self.time_steps):
            if t > 1: # do not update when the timesteps are lower than 1 bc this corresponds to the two initial fields
                self.s.update()
            f = self.s.field.get_val_time(t)
            pp = self.s.particles.list_pos(tstep=t)
            if anim: # create the images for the animation
                (imgname, img) = self.instant_img(f, pp, t, path, dpi)
                list_img.append(img)
                names_img.append(imgname)
            if file: # append the current field to the file
                fstr = Simulation.list2str(f)
                pstr = Simulation.list2str(pp)
                ff.write("{}\n".format(fstr))
                pf.write("{}\n".format(pstr))
            print("{}/{}".format(t, self.time_steps)) if self.log else None
        if anim:
            print("animation finalisation...") if self.log else None
            self.create_gif(list_img, names_img, 60, path, id_img=timestamp)
        if file:
            print("output file finalisation...") if self.log else None

    
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
    
    @staticmethod
    def list2str(l: list):
        """
            Converts a list to a string in a simple format for our use

            :param l: a list to be converted
            :type l: list

            :return: list converted into string
            :rtype: str
        """
        return str(l).replace("[", "").replace("]", "").replace("\n", "")
    
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
    
    def instant_img(self, field: list, particles_pos: list, tstep: int, path: str, dpi: int, ylim=(-0.15, 0.15)):
        """
            Creates an image of the system given the field and a list of the x position of the particles, in the path given, and returns the image opened with Pillow

            :param field: u(x)
            :param particles_pos: position (cell) of each particle
            :param tstep: time step corresponding to the field
            :param path: path for the image to be created
            :param dpi: dpi of the image
            :param ylim: vertical axis interval visualisation

            :type field: list
            :type particles_pos: list
            :type tstep: int
            :type path: str
            :type dpi: int
            :type ylim: tuple

            :return: tuple of the path of the image and the image opened with Pillow
            :rtype: tuple: (str, Pillow object)
        """
        x = np.linspace(0, self.s.length, self.s.nb_linear_steps)
        filename = "{}-{}.png".format(Simulation.IMG_PREFIX, tstep)
        saving_path = "{}\\{}".format(path, filename)
        px = x[particles_pos]
        py = field[particles_pos] # get a list of the y positions of the particle
        
        plt.plot(x, field, "b")
        plt.plot(px, py, "r.")
        plt.title("Field visualisation   t={}s".format(tstep*self.s.dt))
        plt.xlabel("$x$ position on the bench [m]")
        plt.ylabel("$y$ value for the field [m]")
        plt.ylim(ylim)
        plt.savefig(saving_path, dpi=dpi)
        plt.close()

        return (saving_path, Image.open(saving_path))
    
    def create_gif(self, list_images: list, list_names_images: list, fps: int, path: str, id_img=0, del_imgs=True):
        """
            Creates a gif out of the images given

            :param list_images: list of the Pillow images
            :param list_names_images: list of the path+name of the images
            :param fps: frames per second for the gif
            :param path: path for the gif to be created
            :param id_img: suffix at the end of the name of the image (for identification)
            :param del_imgs: delete the images after the creation of the gif

            :type list_images: list
            :type list_names_images: list
            :type fps: int
            :type path: str
            :type id_img: int
            :type del_imgs: bool

            :return: path of the gif created
            :rtype: str
        """
        suffix = "" if id_img == 0 else id_img
        pathgif = "{}\\{}-{}.gif".format(path, Simulation.IMG_PREFIX, suffix)
        list_images[0].save(pathgif, duration=1/fps, save_all=True, append_images=list_images[1:], optimize=True, loop=0)
        if del_imgs:
            for i, fn in zip(list_images, list_names_images):
                i.close()
                os.remove(fn)
        return pathgif


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
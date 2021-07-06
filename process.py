from __future__ import annotations

import json
import os
import datetime
from io import TextIOWrapper

import numpy as np
from matplotlib import pyplot as plt

from field import OneSpaceField
from simulation import Simulation

"""
    Class for loading a simulation and post process the data
"""

class PostProcess:
    ANIM_PREFIX = "QuantumStringANIMATION"
    FOURIER_PREFIX = "FourierTransform"
    SPECTRO_PREFIX = "FourierSpectro"

    def __init__(self, fieldfile: TextIOWrapper, particlesfile: TextIOWrapper, log=False):
        self.log = log
        self.fieldfile = fieldfile
        self.particlesfile = particlesfile
        self.infos = json.loads(fieldfile.readline()) # loads the first line to gather the infos about the simulation
        self.dx = self.infos[Simulation.STR_DX]
        self.dt = self.infos[Simulation.STR_DT]
        self.nx = self.infos[Simulation.STR_SPACESTEPS]
        self.nt = self.infos[Simulation.STR_TIMESTEPS]
        self.L = self.infos[Simulation.STR_LENGTH]
        self.duration = self.dt*self.nt
    
    @staticmethod
    def mean_array(a: list[float], amount: int) -> list[float]:
        mean_size = int(a.size()/amount)
        r = []
        i = 0
    
    def img_field(f: list[float], p: list[int], path=""):
        return path

    def anim(self, path: str):
        ts = int(datetime.datetime.now().timestamp())
        self.fieldfile.seek(0, 0)
        self.particlesfile.seek(0, 0)
        img_files = []
        t = -1
        for field, particles in zip(self.fieldfile, self.particlesfile):
            if t >= 0: # the file has a one-line header (json format)
                field = Simulation.str2list(field, type=float)
                particles = Simulation.str2list(particles, type=int)
                filepath = os.path.join(path, "{}___{}.png".format(ts, t))
                self.img_field(field, particles, path=filepath)
                img_files.append(filepath)
            t += 1
        ### then compiling the images
    
    def fourier(self, *windows, frameskip=1, path=os.path.dirname(os.path.abspath(__file__))):
        ts = int(datetime.datetime.now().timestamp())
        transforms = dict()
        if len(windows) == 0: # if no window given, take the whole string
            windows = [(0.0, 1.0)]
        for w in windows:
            key = str(w)
            transforms[key] = {}
            a, b = int(w[0]*self.nx), int(w[1]*self.nx)
            transforms[key]["window_cells"] = (a, b)
            transforms[key]["mat"] = np.array([[0.0]*(b - a)])
            filename = "{}_{}{}.txt".format(ts, PostProcess.FOURIER_PREFIX, key)
            filepath = os.path.join(path, filename)
            file = open(filepath, "w")
            transforms[key]["file"] = file
            copyinfos = self.infos.copy()
            copyinfos["fourier"] = dict(window_cells=(a, b))
            file.write("{}\n".format(json.dumps(copyinfos)))
        self.fieldfile.seek(0, 0)
        self.next_line(self.fieldfile) # ...
        line = self.next_line(self.fieldfile)
        field = OneSpaceField(line, memory=5)
        frames = 0
        i = 0

        print("processing FFT...") if self.log else None
        while True:
            for tm in transforms.values():
                towrite = ""
                if i % frameskip == 0:
                    a, b = tm["window_cells"]
                    mat = tm["mat"]
                    fft, tm["f"] = field.space_fft(-1, self.infos["dx"], xwindow=(a, b))
                    tm["mat"] = np.append(mat, [fft], axis=0)
                    towrite = Simulation.list2str(fft)
                    frames += 1
                tm["file"].write("{}\n".format(towrite))
            line = self.next_line(self.fieldfile)
            if type(line) == bool:
                break
            field.update(line)
            i += 1
        t = np.linspace(0.0, self.duration, frames)
        
        for key, tm in transforms.items(): # creating the spectrography
            tm["file"].close()
            f = tm["f"]
            mat = tm["mat"]
            ff, tt = np.meshgrid(f, t)
            mat = np.delete(mat, 0, 0)
            plt.pcolormesh(tt, ff, np.abs(mat), shading="gouraud")
            plt.title(key)
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
            plt.savefig(os.path.join(path, "{}-{}-{}.png".format(PostProcess.SPECTRO_PREFIX, key, ts)), dpi=1024)
    
    def next_line(self, file: TextIOWrapper):
        l = file.readline()
        if not l:
            return False
        a = l.split(",")
        try:
            a.remove("\n")
        except:
            pass
        b = [i for i in a if i != ""]
        try:
            b = np.array(b).reshape((1, len(b))).astype(float)
            return b
        except:
            return False
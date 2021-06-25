import json
import os
import time
from io import TextIOWrapper

import numpy as np
from matplotlib import pyplot as plt

from field import OneSpaceField
from simulation import Simulation

class PostProcess:
    IMG_FOURIER_PREFIX = "fft_img___"
    VID_PREFIX = "FourierAnim"

    def __init__(self, fieldfile: TextIOWrapper, log=False):
        self.log = log
        self.fieldfile = fieldfile
        fieldfile.seek(0, 0)
        self.infos = json.loads(fieldfile.readline())
        self.dx = self.infos[Simulation.STR_DX]
        self.dt = self.infos[Simulation.STR_DT]
        self.nx = self.infos[Simulation.STR_SPACESTEPS]
        self.nt = self.infos[Simulation.STR_TIMESTEPS]
        self.L = self.infos[Simulation.STR_LENGTH]
        self.duration = self.dt*self.nt
    
    def fourier(self, *windows, frameskip=1, path=os.path.dirname(os.path.abspath(__file__))):
        transforms = {}
        if len(windows) == 0: # if no window given, take the whole string
            windows = [(0.0, 1.0)]
        for w in windows:
            key = str(w)
            transforms[key] = {}
            a, b = int(w[0]*self.nx), int(w[1]*self.nx)
            transforms[key]["window_cells"] = (a, b)
            transforms[key]["mat"] = np.array([[0.0]*(b - a)])
        self.fieldfile.seek(0, 0)
        self.next_line(self.fieldfile) # ...
        line = self.next_line(self.fieldfile)
        field = OneSpaceField(line, memory=5)
        frames = 0
        i = 0
        newpercent = 0
        img_paths = []
        while True:
            percent = int(i/self.infos["nt"]*100)
            print("{}%".format(percent), end="\r") if self.log and newpercent != percent else None
            newpercent = percent
            if i % frameskip == 0:
                filename = "{}{}.png".format(PostProcess.IMG_FOURIER_PREFIX, frames)
                filepath = os.path.join(path, filename)
                for tm in transforms.values():
                    a, b = tm["window_cells"]
                    mat = tm["mat"]
                    fft, tm["f"] = field.space_fft(-1, self.infos["dx"], xwindow=(a, b))
                    plt.plot(tm["f"], np.abs(fft))
                    plt.savefig(filepath)
                    plt.close()
                    img_paths.append(filepath)
                    tm["mat"] = np.append(mat, [fft], axis=0)
                    frames += 1
            line = self.next_line(self.fieldfile)
            if type(line) == bool:
                break
            field.update(line)
            i += 1
        t = np.linspace(0.0, self.duration, frames)

        Simulation.create_video(img_paths, path, title=PostProcess.VID_PREFIX, log=self.log, compress=False)
        
        for key, tm in transforms.items():
            f = tm["f"]
            mat = tm["mat"]
            ff, tt = np.meshgrid(f, t)
            mat = np.delete(mat, 0, 0)
            plt.pcolormesh(tt, ff, np.abs(mat), shading="gouraud")
            plt.title(key)
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
            plt.show()
    
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

path = "C:/Users/leog/Desktop/lg2021stage/output"
fieldfilename = "QuantumString-field_1624624905.txt"
fieldpath = os.path.join(path, fieldfilename)
ffile = open(fieldpath, "r")
pp = PostProcess(ffile, log=True)
fs = 20 # int(0.001/pp.dt)
pp.fourier((0.0, 0.5), path=path, frameskip=fs)
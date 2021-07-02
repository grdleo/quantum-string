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
    IMG_FOURIER_PREFIX = "fft_img___"
    VID_PREFIX = "FourierAnim"
    SPECTRO_PREFIX = "FourierSpectro"

    def __init__(self, fieldfile: TextIOWrapper, log=False):
        self.log = log
        self.fieldfile = fieldfile
        fieldfile.seek(0, 0)
        self.infos = json.loads(fieldfile.readline()) # loads the first line to gather the infos about the simulation
        self.dx = self.infos[Simulation.STR_DX]
        self.dt = self.infos[Simulation.STR_DT]
        self.nx = self.infos[Simulation.STR_SPACESTEPS]
        self.nt = self.infos[Simulation.STR_TIMESTEPS]
        self.L = self.infos[Simulation.STR_LENGTH]
        self.duration = self.dt*self.nt
    
    def fourier(self, *windows, frameskip=1, path=os.path.dirname(os.path.abspath(__file__))):
        ts = int(datetime.datetime.now().timestamp())
        transforms = {}
        if len(windows) == 0: # if no window given, take the whole string
            windows = [(0.0, 1.0)]
        for w in windows:
            key = str(w)
            transforms[key] = {}
            a, b = int(w[0]*self.nx), int(w[1]*self.nx)
            transforms[key]["window_cells"] = (a, b)
            transforms[key]["mat"] = np.array([[0.0]*(b - a)])
            transforms[key]["img_prefix"] = "{}{}-".format(PostProcess.IMG_FOURIER_PREFIX, key)
            transforms[key]["vid_prefix"] = "{}-{}-".format(PostProcess.VID_PREFIX, key)
            transforms[key]["img_paths"] = []
        self.fieldfile.seek(0, 0)
        self.next_line(self.fieldfile) # ...
        line = self.next_line(self.fieldfile)
        field = OneSpaceField(line, memory=5)
        frames = 0
        i = 0

        print("processing FFT...") if self.log else None
        while True:
            if i % frameskip == 0:
                for tm in transforms.values():
                    filename = "{}{}.png".format(tm["img_prefix"], frames)
                    filepath = os.path.join(path, filename)
                    a, b = tm["window_cells"]
                    mat = tm["mat"]
                    fft, tm["f"] = field.space_fft(-1, self.infos["dx"], xwindow=(a, b))
                    tm["img_paths"].append(filepath)
                    tm["mat"] = np.append(mat, [fft], axis=0)
                    frames += 1
            line = self.next_line(self.fieldfile)
            if type(line) == bool:
                break
            field.update(line)
            i += 1
        t = np.linspace(0.0, self.duration, frames)

        for tm in transforms.values(): # get the max value of the fft so that we fix the scale when plotting
            mat = tm["mat"]
            tm["ymax"] = np.max(np.abs(mat))
        
        for key, tm in transforms.items(): # creating frames and then creating animations for each window
            f = tm["f"]
            ymax = tm["ymax"]
            img_paths = tm["img_paths"]
            tot_frames = len(img_paths)
            vid_prefix = tm["vid_prefix"]
            print("creating frames for {} window".format(key)) if self.log else None
            for fft, img_path, i in zip(tm["mat"], img_paths, range(tot_frames)):
                print("{}/{}".format(i, tot_frames), end="\r") if self.log else None
                plt.plot(f, np.abs(fft))
                plt.ylim(0.0, ymax)
                #plt.yscale("log")
                plt.savefig(img_path, dpi=96)
                plt.close()
            print("done with {} window".format(vid_prefix)) if self.log else None
            Simulation.create_video(img_paths, path, fps=5, title=vid_prefix, log=self.log, compress=True, timestamp=ts)
        
        for key, tm in transforms.items(): # creating the spectrography
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
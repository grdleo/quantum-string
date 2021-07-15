from __future__ import annotations

import json
import os
import datetime
from io import TextIOWrapper

import cv2
import ffmpeg
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits import mplot3d

from quantumstring.field import OneSpaceField
from quantumstring.simulation import Simulation
from quantumstring.particle import Particle, Particles

"""
    Class for loading a simulation and post process the data
"""

class PostProcess:
    ANIM_PREFIX = "QuantumStringANIMATION"
    FOURIER_PREFIX = "FourierTransform"
    SPECTRO_PREFIX = "FourierSpectro"

    COLOR_BLACK = (0, 0, 0)
    COLOR_GRAY = (192, 192, 192)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 0, 0)

    def __init__(self, fieldfile: TextIOWrapper, particlesfile: TextIOWrapper, energyfile: TextIOWrapper, log=False):
        self.log = log

        self.fieldfile = fieldfile
        self.particlesfile = particlesfile
        self.energyfile = energyfile

        self.infos = json.loads(fieldfile.readline()) # loads the first line to gather the infos about the simulation
        self.date = self.infos["date"]
        self.dx = self.infos[Simulation.STR_DX]
        self.dt = self.infos[Simulation.STR_DT]
        self.nx = self.infos[Simulation.STR_SPACESTEPS]
        self.nt = self.infos[Simulation.STR_TIMESTEPS]
        self.L = self.infos[Simulation.STR_LENGTH]
        self.rho = self.infos[Simulation.STR_DENSITY]
        self.T = self.infos[Simulation.STR_TENSION]
        self.left = self.infos[Simulation.STR_EDGE_LEFT]
        self.right = self.infos[Simulation.STR_EDGE_RIGHT]
        self.duration = self.dt*self.nt
        self.particles = self.infos["particles"]

        self.c = np.sqrt(self.T/self.rho)
    
    def energy(self):
        self.energyfile.seek(0, 0)
        nrj_tot = []

        t = -1
        for energyline in self.energyfile:
            if t >= 0:
                e = Simulation.str2list(energyline)
                nrj_tot.append(np.sum(e)*self.dx)
            t += 1
        
        tline = np.linspace(0.0, self.duration, self.nt)

        ax = plt.axes()
        ax.plot(tline, nrj_tot, "b.")
        ax.set_title("Total energy of the string (simulation {})".format(self.date))
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Energy [J]")
        plt.show()
    
    def plot_particles(self) -> None:
        pp = self.particles_pos()
        tline = np.linspace(0, self.duration, self.nt)
        ax = plt.axes()
        for pos, part in zip(pp.T, self.particles):
            c = tuple(np.array(part[Particle.STR_COLOR])/255.0)
            ax.plot(tline, pos, color=c)
        ax.set_title("Particles vertical position (simulation {})".format(self.date))
        ax.set_xlabel("t [s]")
        ax.set_ylabel("z [m]")
        plt.show()
        
    def particles_pos(self) -> np.ndarray:
        """
            r[t,n] where r is the position of the particle, t the timestep considered, and n the index of the particle
        """
        self.fieldfile.seek(0, 0)
        self.particlesfile.seek(0, 0)
        r = []

        t = -1
        for field, particles in zip(self.fieldfile, self.particlesfile):
            if t >= 0:
                field = Simulation.str2list(field, type=float)
                particles = Simulation.str2list(particles, type=int)
                pos_part = []
                for pos in particles:
                    pos_part.append(field[pos])
                pos_part = np.array(pos_part)
                r.append(pos_part)
            t += 1
        return np.array(r)
    
    def img_field(self, baseimg: np.ndarray, f: list[float], p: list[int], anim_params: dict, timestep: int, yscale=1.0) -> np.ndarray:
        """
            Create and returns an image of the current state of the simulation

            :param baseimg: base PIL Image to write onto
            :param field: state of the string
            :param particles_pos: the position (cell) of each particles
            :param tstep: time step corresponding to the state
            :param yscale: scaling factor for vertical axis
            :param infos: if True, prompt information of the simulation on video
        """
        img = np.copy(baseimg)
        x = np.linspace(anim_params["ox"], anim_params["endstring"], f.size).astype(np.int32)
        y = (-anim_params["ppx_per_m"]*yscale*f + anim_params["oy"]).astype(np.int32)
        string = np.array([(x[i], y[i]) for i in range(1, f.size - 1)])
        string = string.reshape((-1, 1, 2))

        cv2.polylines(img, [string], False, PostProcess.COLOR_WHITE)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "t={:.6f}s".format(self.dt*timestep), (2, anim_params["ly"] - 5), font, 0.5, PostProcess.COLOR_GRAY, 1, cv2.LINE_AA) 

        r = int(anim_params["mass_rad"])
        for pidx, part in zip(p, self.particles):
            center = (int(x[pidx]), int(y[pidx]))
            cv2.circle(img, center, r, part[Particle.STR_COLOR], -1)

        img[y[0], x[0]] = PostProcess.COLOR_GRAY
        img[y[len(y)-1], x[len(x)-1]] = PostProcess.COLOR_GRAY
        
        return img

    def anim(self, path: str, title=False, resolution=(720, 480), fps=60, frameskip=1, yscale=1.0, log=True, compress=True):
        ts = int(datetime.datetime.now().timestamp())
        self.fieldfile.seek(0, 0)
        self.particlesfile.seek(0, 0)

        title = PostProcess.ANIM_PREFIX if type(title) == bool and not title else title
        videopath = os.path.join(path, "{}-{}-UNCOMPRESSED.mp4".format(title, ts))
        videopath_compressed = os.path.join(path, "{}-{}.mp4".format(title, ts))

        lx, ly = resolution[0], resolution[1]
        margin_ppx = 5
        length_ppx = lx - 2*margin_ppx
        anim_params = dict(
            lx=lx,
            ly=ly,
            ox=margin_ppx,
            oy=0.5*ly,
            endstring=margin_ppx + length_ppx,
            ppx_per_m=length_ppx,
            mass_rad=3
        )

        baseimg = np.zeros((resolution[1], resolution[0], 3), np.uint8)

        textinfos = "L={:.2f}m ρ={:.2f}g/m T={:.2f}N c={:.2f}m/s\n{} ~ {}".format(
            self.L, 
            self.rho*1e3, 
            self.T, 
            np.sqrt(self.T/self.rho),
            self.left,
            self.right
            )
        
        fontpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Roboto-Regular.ttf")
        font = ImageFont.truetype(fontpath, 16)
        img_pil = Image.fromarray(baseimg)
        draw = ImageDraw.Draw(img_pil)
        draw.multiline_text((2, 2), textinfos, font=font, fill=PostProcess.COLOR_GRAY)
        lineh = 12
        hcount = 0
        for p in self.particles:
            textparticle = "m={:.1f}g, ω={:.1f}rad/s".format(p[Particle.STR_MASS]*1e3, p[Particle.STR_PULSATION])
            c = tuple(p[Particle.STR_COLOR])
            draw.text((lx - 200, 2 + hcount), textparticle, font=font, fill=c)
            hcount += lineh
        baseimg = np.array(img_pil)

        video = cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (lx, ly))

        print("Animation for {} simulation:".format(self.date))
        t = -1
        for field, particles in zip(self.fieldfile, self.particlesfile):
            if t >= 0: # the file has a one-line header (json format)
                print("{}/{} images processed".format(int(t/frameskip), int(self.nt/frameskip)), end="\r") if log else None
                field = Simulation.str2list(field, type=float)
                particles = Simulation.str2list(particles, type=int)
                if t%frameskip == 0:
                    img = self.img_field(baseimg, field, particles, anim_params, t, yscale=yscale)
                    video.write(img)
            t += 1
        
        video.release()
        cv2.destroyAllWindows()

        return_path = videopath
        if compress:
            try:
                video = ffmpeg.input(videopath)
                video = ffmpeg.output(video, videopath_compressed, vcodec="h264")
                ffmpeg.run(video)
                os.remove(videopath)
                return_path = videopath_compressed
            except:
                print("WARNING: could not compress the video due to 'ffmpeg' error...") if log else None

        print("video created successfully!") if log else None
        return return_path
    
    @staticmethod
    def reduce_axis(mat: np.ndarray, factor: int, axis=0):
        """
            Given a matrix (an×m) or (n×am), returns a matrix (n×m) where the cells are the means of the cells surroundings

            ex: 
            >>> mat = np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]
            ])
            >>> PostProcess.reduce_axis(mat, 2, axis=0)
            >>> np.array([
                [0+4, 1+5, 2+6, 3+7],
                [8+12, 9+13, 10+14, 11+15]
            ])/2
            >>> matt = PostProcess.reduce_axis(mat, 2, axis=1)
            >>> matt
            >>> np.array([
                [0+1, 2+3],
                [4+5, 6+7],
                [8+9, 10+11],
                [12+13, 14+15]
            ])/2
        """
        complementary = 1 if axis == 0 else 0
        s = mat.shape[complementary]
        q = mat.shape[axis]

        if s%factor != 0:
            raise ValueError("Error with matrix shape: {} is not a multiple of {}".format(s, factor))

        l = np.eye(int(q), dtype=float)
        r = np.repeat(np.eye(int(s/factor), dtype=float), repeats=factor, axis=axis)
        if axis:
            l, r = r, l
        
        return l.dot(mat).dot(r)/factor
    
    @staticmethod
    def prime_factors(num: int) -> list[int]:  
        primes = []
        # Using the while loop, we will print the number of two's that divide n  
        while num%2 == 0:  
            primes.append(2) 
            num = int(num*0.5)
    
        for i in range(3, int(np.sqrt(num)) + 1, 2):  
            # while i divides n , print i ad divide n  
            while num%i == 0:  
                primes.append(i)
                num = int(num/i)  
        
        if num > 2:
            primes.append(num)

        return primes

    @staticmethod
    def all_subsets(*el: object) -> list:
        n = len(el)
        subsets = []
        for i in range(0, 2**n):
            bin_str = bin(i).replace("0b", "").zfill(n)
            current = []
            for b, idx in zip(bin_str, range(0, len(bin_str))):
                b = int(b)
                if b:
                    current.append(el[idx])
            subsets.append(current)
        return subsets

    @staticmethod
    def reduce_space(mat: np.ndarray, ideal_size: tuple[int]) -> np.ndarray:
        """
            Reduces a given matrix doing the mean method, by a factor close to the one given in argument. If no factor is given, the highest possible will be made
        """
        for i, ideal, axis in zip(mat.shape, ideal_size, (1, 0)):
            primes = PostProcess.prime_factors(i)[0:16] # we take no more than 16 primes, otherwise the subsets will be too long to compute...
            subsets = PostProcess.all_subsets(*primes)
            all_factors = []
            for sub in subsets:
                nb = np.prod(sub)
                all_factors.append(nb)
            all_factors = np.array(all_factors)
            factor = all_factors[np.abs(i/all_factors - ideal).argmin()]
            mat = PostProcess.reduce_axis(mat, factor, axis=axis) if i > ideal else mat

        return mat
    
    def get_field(self, matrix_ideal_res: int) -> tuple[np.ndarray]:
        self.fieldfile.seek(0, 0)
        Z = [] # mat[t,x]
        t = -1
        for f in self.fieldfile:
            if t >= 0:
                Z.append(Simulation.str2list(f))
            t =+ 1
        Z = np.array(Z)

        ideal_size = (matrix_ideal_res, matrix_ideal_res) # Z matrix is often too large to be plot correctly...
        Z = PostProcess.reduce_space(Z, ideal_size) if matrix_ideal_res < np.inf else Z
        tsize, xsize = Z.shape

        xline = np.linspace(0.0, self.L, xsize)
        tline = np.linspace(0.0, self.duration*self.c, tsize)
        X, T = np.meshgrid(xline, tline)

        return X.T, T.T, Z.T

    def plot3d(self, matrix_ideal_res=128, cmap="viridis"):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(*self.get_field(matrix_ideal_res),
                rstride=1, cstride=1,
                cmap=cmap, edgecolor='none')
        ax.set_title("Graph evolution of the field (simulation {})".format(self.date))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("ct [m]")
        ax.set_zlabel("u [m]")
        plt.show()
    
    def plot2d(self, matrix_ideal_res=np.inf, cmap="viridis"):
        fig = plt.figure()
        ax = plt.axes()
        im = ax.pcolormesh(*self.get_field(matrix_ideal_res),
                cmap=cmap,
                rasterized=True)
        ax.set_title("Color mesh of the field (simulation {})".format(self.date))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("ct [m]")
        fig.colorbar(im, ax=ax)
        plt.show()

    def fourier(self, *windows, frameskip=1, path=os.path.dirname(os.path.abspath(__file__))):
        ts = int(datetime.datetime.now().timestamp())
        self.fieldfile.seek(0, 0)
        self.particlesfile.seek(0, 0)

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

        osf = OneSpaceField(np.array([[0.0]*self.nx]), memory=3)
        frames = 0
        t = -1

        print("processing FFT...") if self.log else None
        for line in self.fieldfile:
            if t >= 0: # the file has a one-line header (json format)
                field = Simulation.str2list(line)
                osf.update(field)
                for tm in transforms.values():
                    towrite = ""
                    if t%frameskip == 0:
                        a, b = tm["window_cells"]
                        mat = tm["mat"]
                        fft, tm["f"] = osf.space_fft(-1, self.infos["dx"], xwindow=(a, b))
                        tm["mat"] = np.append(mat, [fft], axis=0)
                        towrite = Simulation.list2str(fft)
                    tm["file"].write("{}\n".format(towrite))
                    frames = tm["mat"].shape[0] - 1
            t += 1

        time = np.linspace(0.0, self.duration, frames)
        
        for key, tm in transforms.items(): # creating the spectrography
            tm["file"].close()
            f = tm["f"]
            mat = tm["mat"]
            ff, tt = np.meshgrid(f, time)
            mat = np.delete(mat, 0, 0)
            plt.pcolormesh(tt, ff, np.abs(mat), shading="gouraud")
            plt.title(key)
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
            plt.savefig(os.path.join(path, "{}-{}-{}.png".format(PostProcess.SPECTRO_PREFIX, key, ts)), dpi=1024)
from phystring import PhyString
from particle import Particle, Particles
from edge import Edge, MirrorEdge

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import ffmpeg

import os
import math
import time
import datetime
import json

class Simulation:
    """
        Class that wraps the whole simulation thing
    """
    IMG_PREFIX = "qs_img___"
    VID_PREFIX = "QuantumString"
    NB_ZEROS = 8
    IMG_FORMAT = "png"
    PERCENT_MAX = 256

    STR_DT = "dt"
    STR_DX = "dx"
    STR_TIMESTEPS = "nt"
    STR_SPACESTEPS = "nx"
    STR_TENSION = "T"
    STR_DENSITY = "rho"
    STR_LENGTH = "L"

    def __init__(self, dt: float, time_steps: int, string_discret: int, string_len: float, string_density: float, string_tension: float, edge_left: Edge, edge_right: Edge, ic_pos: list, ic_vel: list, particles, memory_field=5, log=True):
        """
            Initialisation of the simulation

            :param dt: value of the time step [s]
            :param time_steps: number of time steps
            :param string_discret: number of cells in the string
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

            :type dt: float
            :type time_steps: int
            :type string_discret: int
            :type string_len: float
            :type string_c: float
            :type string_density: float
            :type edge_left: Edge class
            :type edge_right: Edge class
            :type ic_pos: list
            :type ic_vel: list
            :type particles: Particles object
            :type memory_field: int
            :type log: bool

        """
        self.log = log
        self.time_steps = time_steps
        self.dt = dt
        self.time = str(datetime.datetime.now())

        self.s = PhyString(string_len, string_discret, dt, string_density, string_tension, edge_left, edge_right, ic_pos, ic_vel, particles, memory_field=memory_field)
    
    def infos(self):
        return {
            "desc": "QUANTUM STRING SIMULATION",
            "date": self.time,
            Simulation.STR_DT: self.s.dt,
            Simulation.STR_DX: self.s.dx,
            Simulation.STR_TIMESTEPS: self.time_steps,
            Simulation.STR_SPACESTEPS: self.s.nb_linear_steps,
            Simulation.STR_LENGTH: self.s.length,
            Simulation.STR_TENSION: self.s.tension,
            Simulation.STR_DENSITY: self.s.linear_density,
            "particles": self.s.particles.infos()
        }
    
    def __repr__(self):
        return "[SIMULATION]    Δt={}s, Δx={}m, time steps={}, string steps (nb discretisation)={}\n{}\n{}".format(
            self.s.dt, 
            self.s.dx, 
            self.time_steps, 
            self.s.nb_linear_steps,
            self.s,
            self.s.particles)

    def img_file_path(self, path: str, i: int):
        nb_str = str(i).zfill(Simulation.NB_ZEROS)
        return "{}\\{}{}.{}".format(path, Simulation.IMG_PREFIX, nb_str, Simulation.IMG_FORMAT)

    def run(self, path: str, anim=True, file=True, res=(480, 320), frameskip=True, yscale=5.0, window_anim=False) -> tuple:
        """
            Runs the simulation with options to save it as a animation and/or in a file

            :param path: path for the simulation to be saved
            :type path: str

            :return: a tuple containing the path for the field file and the particles file
            :rtype: tuple
        """
        dtnow = datetime.datetime.now()
        timestamp = int(dtnow.timestamp())
        jsoninfos = json.dumps(self.infos())
        print(self) if self.log else None

        field_file_path = os.path.join(path, "QuantumString-field_{}.txt".format(timestamp))
        particles_file_path = os.path.join(path, "QuantumString-particles_{}.txt".format(timestamp))

        if file:
            begtxt = "{}\n".format(jsoninfos)
            ff = open(field_file_path, "w", encoding="utf-8")
            pf = open(particles_file_path, "w", encoding="utf-8")
            ff.write(begtxt)
            pf.write(begtxt)
        
        cblack = (0, 0, 0)
        template_anim = Image.new('RGB', res, color=cblack)

        render_len = self.s.length
        render_nbcells = self.s.nb_linear_steps
        render_cellstart, render_cellstop = 0, render_nbcells
        if anim:
            if window_anim != False:
                a, b = window_anim[0], window_anim[1]
                render_cellstart, render_cellstop = int(a*render_len/self.s.dx), int(b*render_len/self.s.dx)
                render_nbcells = render_cellstop - render_cellstart
                render_len = render_nbcells*self.s.dx
                
            pairoddlist = [i for i in range(0, 2*render_nbcells)]
            pairoddlist = np.array(pairoddlist)
            self.anim_params = {
                "max_frames": 1500,
                "margin": 0,
                "mass_rad": 3,
                "wh": res,
                "linx": np.linspace(0.0, render_len, render_nbcells),
                "forx": pairoddlist % 2 == 0,
                "fory": pairoddlist % 2 == 1,
                "origin": None,
                "pix_per_m": None
            }
            self.anim_params["origin"] = (self.anim_params["margin"], int(0.5*res[1]))
            self.anim_params["pix_per_m"] = res[0]/render_len

            if type(frameskip) == bool and frameskip:
                max_frames = self.anim_params["max_frames"]
                tot_frames = max_frames if self.time_steps >= max_frames else self.time_steps
                frameskip = int(self.time_steps/tot_frames)
            else:
                frameskip = 1
        list_imgs = []

        percent = 0
        ts = datetime.datetime.now()
        list_dt_compute = []
        for t in range(0, self.time_steps):
            if t > 1: # do not update when the timesteps are lower than 1 bc this corresponds to the two initial fields
                self.s.update() # update the string
            
            prop = t/self.time_steps
            newpercent = math.floor(prop*Simulation.PERCENT_MAX)
            if (newpercent != percent) and self.log: # update the console
                newts = datetime.datetime.now()
                dtcompute = (newts - ts).total_seconds()
                elapsed = sum(list_dt_compute)
                list_dt_compute.append(dtcompute)
                spinner = "←↖↑↗→↘↓↙" # ".ₒoO0Ooₒ"
                load = percent % len(spinner)
                print("{:2}% {} {:.4f}s left                ".format(int(percent/Simulation.PERCENT_MAX*100), spinner[load:load+1],  float(elapsed*(1/prop-1))), end="\r")
                ts = newts
            percent = newpercent
            
            f = self.s.field.get_val_time(t)
            pp = self.s.particles.list_pos(tstep=t)
            if anim: # create the images for the animation
                if t % frameskip == 0:
                    f_render = f[render_cellstart:render_cellstop]
                    pp_render = pp - render_cellstart
                    img = self.instant_img(template_anim.copy(), f_render, pp_render, t, yscale=yscale)
                    thisimgpath = self.img_file_path(path, t)
                    img.save(thisimgpath)
                    list_imgs.append(thisimgpath)
            if file: # append the current field to the file
                fstr = Simulation.list2str(f)
                pstr = Simulation.list2str(pp)
                ff.write("{}\n".format(fstr))
                pf.write("{}\n".format(pstr))
        print("")
        
        if anim:
            print("video output creation...") if self.log else None
            Simulation.create_video(list_imgs, path, title=Simulation.VID_PREFIX, log=self.log, timestamp=timestamp)
        if file:
            return field_file_path, particles_file_path
    @staticmethod
    def list2str(l: list):
        """
            Converts a list to a string in a simple format for our use

            :param l: a list to be converted
            :type l: list

            :return: list converted into string
            :rtype: str
        """
        return str(list(l)).replace("[", "").replace("]", "").replace(" ", "").replace("\n", "")
    
    def instant_img(self, baseimg: Image, field: list, particles_pos: list, tstep: int, yscale=1.0) -> Image:
        """
            Creates an image of the current state of the simulation

            :param baseimg: base PIL Image to write onto
            :param field: state of the string
            :param particles_pos: the position (cell) of each particles
            :param tstep: time step corresponding to the state
            :param yscale: scaling factor for vertical axis

            :return: Pillow image of the current state
            :rtype: PIL.Image
        """
        d = ImageDraw.Draw(baseimg)
        mass_rad = self.anim_params["mass_rad"]
        (ox, oy) = self.anim_params["origin"]
        pix_per_m = self.anim_params["pix_per_m"]
        linx = self.anim_params["linx"]
        forx = self.anim_params["forx"]
        fory = self.anim_params["fory"]
        line_vals = np.zeros(shape=2*field.shape[-1]).astype(int)
        px = (linx*pix_per_m + ox).astype(int)
        py = (-field*pix_per_m*yscale + oy).astype(int)
        line_vals[forx] = px
        line_vals[fory] = py
        begs = px[0]
        ends = px[-1]
        line_vals = list(line_vals)
        ctxt = (255, 0, 0)
        crep = (0, 0, 255)
        cgray = (64, 64, 64)
        cwhite = (255, 255, 255)
        cm_in_pix = pix_per_m*0.01
        cm_scaled = int(cm_in_pix*yscale)
        d.line([begs, oy, ends, oy], fill=cgray) #horizontal
        d.line(line_vals, fill=cwhite) # string
        d.line([ends, oy - cm_scaled, ends, oy + cm_scaled], fill=crep) # scale y line
        d.line([ends, oy + cm_scaled, ends - cm_in_pix, oy + cm_scaled], fill=crep) # unscale x line
        d.text((ends - 15, oy - cm_scaled - 15), "1cm", fill=crep) # +text scale
        d.multiline_text((2, 2), "{}\n{}".format(self.s, str(self.s.particles).replace(";", "\n")), fill=ctxt)
        d.text((baseimg.size[0] - 70, baseimg.size[1] - 30), "t={:0<6}".format(round(tstep*self.dt, 6)), fill=ctxt)
        for p in particles_pos:
            if 0 <= p < len(linx):
                pos = (px[p], py[p])
                d.ellipse([pos[0] - mass_rad, pos[1] - mass_rad, pos[0] + mass_rad, pos[1] + mass_rad], fill=(255, 0, 0))
        return baseimg

    @staticmethod
    def create_video(list_paths: list, output_path: str, compress=True, title="", fps=60, log=False, timestamp=datetime.datetime.now()):
        """
            from a list of image paths, create a video out of it
        """
        frame = cv2.imread(list_paths[0])
        height, width, layers = frame.shape
        videopath = os.path.join(output_path, "{}-{}-UNCOMPRESSED.mp4".format(title, timestamp))
        videopath_compressed = os.path.join(output_path, "{}-{}.mp4".format(title, timestamp))
        video = cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
        total_frames = len(list_paths)
        for img, i in zip(list_paths, range(0, total_frames)):
            print("{}/{} images computed".format(i, total_frames), end="\r") if log else None
            video.write(cv2.imread(img))
            for r in range(64):
                try:
                    os.remove(img)
                    break
                except:
                    pass
        cv2.destroyAllWindows()
        video.release()
        if compress:
            video = ffmpeg.input(videopath)
            video = ffmpeg.output(video, videopath_compressed, vcodec="h264")
            ffmpeg.run(video)
            os.remove(videopath)

class RestString(Simulation):
    """
        Abstraction of Simulation: the initial field is at rest
    """
    def __init__(self, dt: float, time_steps: int, string_discret: int, string_len: float, string_density: float, string_tension: float, edge_left: Edge, edge_right: Edge, particles, log=True, memory_field=5):
        ic_pos = [0]*string_discret
        ic_vel = ic_pos.copy()
        super().__init__(dt, time_steps,  string_discret, string_len, string_density, string_tension, edge_left, edge_right, ic_pos, ic_vel, particles, log=log, memory_field=memory_field)

class FreeString(RestString):
    """
        Abstraction of RestString: the system is particle free
    """
    def __init__(self, dt: float, time_steps: int, string_discret: int, string_len: float, string_density: float, string_tension: float, edge_left: Edge, edge_right: Edge, log=True, memory_field=5):
        particles = Particles(string_discret, [])
        super().__init__(dt, time_steps, string_discret, string_len, string_density, string_tension, edge_left, edge_right, particles, log=log, memory_field=memory_field)

class CenterFixed(RestString):
    """
        Abstraction of RestString: the system has a single particle in the center of the string
    """
    def __init__(self, dt: float, time_steps: int, string_discret: int, string_len: float, string_density: float, string_tension: float, edge_left: Edge, edge_right: Edge, mass_particle: float, pulsation_particle: float, log=True, memory_field=5):
        center_string = math.floor(string_discret*0.5)
        p = Particle(center_string, 0.0, mass_particle, pulsation_particle, True, string_discret)
        particles = Particles(string_discret, [p])
        super().__init__(dt, time_steps, string_discret, string_len, string_density, string_tension, edge_left, edge_right, particles, log=log, memory_field=memory_field)

class Cavity(Simulation):
    """
        Abstraction of Simulation: mirrors in both ends, initial position given but initial velocity = 0
    """
    def __init__(self, dt: float, time_steps: int, string_discret: int, string_len: float, string_density: float, string_tension: float, ic_pos: list, particles: Particles, log=True, memory_field=5):
        ml, mr = MirrorEdge(), MirrorEdge()
        ic_pos = np.array(ic_pos)
        ic_vel = [0]*len(ic_pos)
        ic_vel = np.array(ic_vel)
        super().__init__(dt, time_steps, string_discret, string_len, string_density, string_tension, ml, mr, ic_pos, ic_vel, particles, log=log, memory_field=memory_field)
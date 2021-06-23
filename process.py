import json
import os
import time

import numpy as np
from matplotlib import pyplot as plt

from field import OneSpaceField

path = "C:/Users/leog/Desktop/lg2021stage/output"
fieldfilename = "QuantumString-field_1624452111.txt"
fieldpath = os.path.join(path, fieldfilename)
fieldfile = open(fieldpath, "r")
infos = json.loads(fieldfile.readline())
duration = infos["dt"]*infos["nt"]

def next_line(file):
    l = file.readline()
    if not l:
        return False
    a = l.split(",")
    try:
        a.remove("\n")
    except:
        pass
    b = [i for i in a if i != ""]
    b = np.array(b).reshape((1, len(b))).astype(float)
    return b

line = next_line(fieldfile)
field = OneSpaceField(line)
f = []
t = np.linspace(0.0, duration, infos["nt"])
mat = np.array(line)
while True:
    fft, f = field.space_fft(-1, infos["dx"])
    mat = np.vstack((mat, np.array([fft])))
    line = next_line(fieldfile)
    if type(line) == bool:
        break
    field.update(line)
ff, tt = np.meshgrid(f, t)
mat = np.delete(mat, 0, 0)
plt.pcolormesh(tt, ff, np.abs(mat), shading="gouraud")
plt.show()
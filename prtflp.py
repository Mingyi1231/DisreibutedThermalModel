#-- print each chiplet's center coordinate from a hotspot flp file

import os
import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


if len(sys.argv) != 2:
    print("Usage: python3 prtflp <flp_file>")
    sys.exit(1)


#-- read floorplan file
#-- format: unitname\tdx\tdy\tx0\ty0
def readflpfile(file):
    entries = list(csv.reader(open(file, 'r'), delimiter='\t'))
    return entries


entries = readflpfile(sys.argv[1])


max_x = 0
max_y = 0
i = 0
for e in entries:
    #-- skip blank lines
    if len(entries[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if entries[i][0][0] == '#':
        i += 1
        continue
    name = entries[i][0]
    left, width = float(entries[i][3]), float(entries[i][1])
    bottom, height = float(entries[i][4]), float(entries[i][2])
    right = left + width
    top = bottom + height
    if max_x < right:
        max_x = right
    if max_y < top:
        max_y = top
    i += 1


total_width = max_x
total_length = max_y


# build a rectangle for each unit in axes coords
i = 0
for e in entries:
    #-- skip blank lines
    if len(entries[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if entries[i][0][0] == '#':
        i += 1
        continue
    name = entries[i][0]
    left, width = float(entries[i][3]), float(entries[i][1])
    bottom, height = float(entries[i][4]), float(entries[i][2])
    right = left + width
    top = bottom + height

    print("%s: [%f  %f]" % (entries[i][0], left+0.5*width, bottom+0.5*height))
    i += 1


#-- chiplet count
chiplet_count = 0
i = 0
xc = np.array([])
yc = np.array([])
for e in entries:
    #-- skip blank lines
    if len(entries[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if entries[i][0][0] == '#':
        i += 1
        continue

    if entries[i][0][0] == 'C':
        left, width = float(entries[i][3]), float(entries[i][1])
        bottom, height = float(entries[i][4]), float(entries[i][2])
        xc = np.append(xc, left+0.5*width)
        yc = np.append(yc, bottom+0.5*height)
        chiplet_count += 1
    i += 1


print("Distances from Chiplet0")
for j in range(0, chiplet_count):
    dx = xc[j] - xc[0]
    dy = yc[j] - yc[0]
    dd = math.sqrt(dx*dx + dy*dy)
    print("Chiplet0-Chiplet%d: %f" % (j, dd))


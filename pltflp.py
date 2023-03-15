#-- plot floorplan from a hotspot flp file

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


if len(sys.argv) != 2:
    print("Usage: python3 pltflp <flp_file>")
    sys.exit(1)


#-- read floorplan file
#-- format: unitname\tdx\tdy\tx0\ty0
def readflpfile(file):
    entries = list(csv.reader(open(file, 'r'), delimiter='\t'))
    '''
    fp = open(file)
    rdr = csv.DictReader(filter(lambda row: row[0] != '#', fp))
    entries = list(row)
    '''
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

#-- define figure and axis
fig, ax = plt.subplots()


#ax = fig.add_axes([0, 0, 1.1*max_x, 1.1*max_y])
#ax.plot([0,0.01], [0,0.01])

#-- alpha: transparency: 1=fully visible, 0=invisible
ax.plot([0,max_x], [0,max_y], color='white', alpha=0.0)
#ax.plot([0,max_x], [0,max_y], color='white', alpha=1)


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

    #ax.add_patch(Rectangle((0.0002, 0.0002), 0.0004, 0.0004))
    ax.add_patch(Rectangle((left, bottom), width, height,
        fill=False, clip_on=False,
        edgecolor='blue', facecolor='blue', lw=3))
    ax.text(0.5*(left+right), 0.5*(bottom+top), name,
        fontsize=10, color='red')

    '''
    ax.text(0.5*(left+right), 0.5*(bottom+top), name,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=20, color='red',
        transform=ax.transAxes)
    '''
    i += 1

#ax.set_axis_off()

ax.set_xticks([n for n in np.linspace(0, total_width, 5)])
ax.set_xticklabels([n*(10**3) for n in np.linspace(0, total_width, 5)])
ax.set_xlabel("Horizontal Position (mm)")

ax.set_yticks([n for n in np.linspace(0, total_length, 5)])
ax.set_yticklabels([n*(10**3) for n in np.linspace(0, total_length, 5)])
ax.set_ylabel("Vertical Position (mm)")


plt.show()

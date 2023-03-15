#-- plot hotspot temperature map with floorplan
#-- NOTE: only python2 can run this code
#-- NOTE: python3.9.7 can also run this code

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


if len(sys.argv) != 3:
    print("Usage: python3 pltflptemp.py <floorplan_file> <steady_grid_file>")
    sys.exit(1)


#-- read in csv format voltage file 
#-- format: xloc\tyloc\tvoltage
def ReadCSVfile(filename):
    array = np.genfromtxt(filename)
    return array

#-- read floorplan file
#-- format: unitname\tdx\tdy\tx0\ty0
def readflpfile(file):
    blocks = list(csv.reader(open(file, 'r'), delimiter='\t'))
    '''
    fp = open(file)
    rdr = csv.DictReader(filter(lambda row: row[0] != '#', fp))
    blocks = list(row)
    '''
    return blocks


blocks = readflpfile(sys.argv[1])


max_x = 0
max_y = 0
i = 0
for e in blocks:
    #-- skip blank lines
    if len(blocks[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if blocks[i][0][0] == '#':
        i += 1
        continue
    name = blocks[i][0]
    left, width = float(blocks[i][3]), float(blocks[i][1])
    bottom, height = float(blocks[i][4]), float(blocks[i][2])
    right = left + width
    top = bottom + height
    if max_x < right:
        max_x = right
    if max_y < top:
        max_y = top
    i += 1


total_width = max_x
total_height = max_y

res_array = ReadCSVfile(sys.argv[2])
npts = len(res_array)
#gc = res_array[:,0]
#z = res_array[:,1] - 273.15
#z = res_array[:,1]
dx = res_array[:,0]
dy = res_array[:,1]
z = res_array[:,2]

#-- print statistics
print("Total grid count: %d" %z.size)
print("Min grid resistance: %f[C]" % np.min(z))
print("Avg grid resistance: %f[C]" % np.average(z))
print("Max grid resistance: %f[C]" % np.max(z))

fig,(ax) = plt.subplots(nrows=1)
#-- ax = axes[2,2], subplots 1 row only, ax = axed[2,2] is a usage example

#x_grid_count = 64
#y_grid_count = 64
## x_grid_count = y_grid_count = np.sqrt(np.len(grid_idx))
#
#
##-- NOTE: hotspot's grid index is computed from top left corner
#x = np.array([])
#y = np.array([])
#for i in range(y_grid_count):
#    for j in range(x_grid_count):
#        x = np.append(x, j*max_x/x_grid_count)
#        y = np.append(y, total_height - i*max_y/y_grid_count)

# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

ax.tricontour(dx, dy, z, levels=14, linewidths=0.5, colors='k')
cntr2 = ax.tricontourf(dx, dy, z, levels=14, cmap="RdBu_r")



# build a rectangle for each unit in axes coords
i = 0
for e in blocks:
    #-- skip blank lines
    if len(blocks[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if blocks[i][0][0] == '#':
        i += 1
        continue
    name = blocks[i][0]
    left, width = float(blocks[i][3]), float(blocks[i][1])
    bottom, height = float(blocks[i][4]), float(blocks[i][2])
    right = left + width
    top = bottom + height

    ax.add_patch(Rectangle((left, bottom), width, height,
        fill=False, clip_on=False,
        edgecolor='black', facecolor='blue', lw=1))
    ax.text(0.5*(left+right), 0.5*(bottom+top), name,
        fontsize=10, color='black')
    i += 1


fig.colorbar(cntr2, ax=ax)
ax.plot(dx, dy, 'ko', ms=3)
ax.set_title('Distributed thermal resistance contour (%s)' % sys.argv[1])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


plt.subplots_adjust(hspace=0.5)
plt.show()


##-- plot temperature histogram
#n_bins = 20
#
#print("No. of nodes: %d" % len(z))
#
##-- Creating histogram
#fig, axs = plt.subplots(1, 1)
#
#axs.hist(z, bins = n_bins)
#
#axs.set_title('Temperature histogram (%s)' % sys.argv[1])
#axs.set_xlabel("Temperature [C]")
#axs.set_ylabel("Grid count")

#-- Show plot
plt.show()


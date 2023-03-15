import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

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

path = './outputs/case2/'
blocks = readflpfile(path + "table_modelL4_ChipLayer.flp")

array1 = ReadCSVfile(path + 'step_999.grid.steady')
array2 = ReadCSVfile(path + 'table_model.grid.steady')
z1 = array1[:,1] - 273.15 
z2 = array2[:,1] - 273.15 
num1 = len(z1)
num2 = len(z2)
assert(num1 == num2)
#for i in range(num2):
#    z2[i] = float(array2[i]) - 45

#array_sum = [0]*num
diff = np.array([])
absdiff = np.array([])
for i in range(num1):
    diff = np.append(diff, (z1[i] - z2[i]))
    absdiff = np.append(absdiff, abs(z1[i] - z2[i]))


#    if diff[i]<0.005:
#        count1 += 1
#    if diff[i]<0.05:
#        count2 += 1
#    if diffmax<diff[i]:
#        diffmax = diff[i]
#        countmax = i
#    if diffmin>diff[i]:
#        diffmin = diff[i]
#        countmin = i
##    if z1[i] <= 1:
##        count += 1
#print("gridtemp_diffmin:",diffmin)
#print("gridtemp_diffmax:",diffmax)
##print("seqmax",countmax)
##print("seqmin",countmin)
#
#print("diff<0.005C grid count:",count1)
#print("diff<0.05C grid count:",count2)

fig,(ax) = plt.subplots(nrows=1)

x = np.array([])
y = np.array([])

x_grid_count = 64
y_grid_count = 64
max_x = 1e-3*50
max_y = 1e-3*50
total_width = max_x
total_height = max_y

for i in range(y_grid_count):
    for j in range(x_grid_count):
        x = np.append(x, j*max_x/x_grid_count)
        y = np.append(y, total_height - i*max_y/y_grid_count)

# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

ax.tricontour(x, y, absdiff, levels=14, linewidths=0, colors='k')
cntr2 = ax.tricontourf(x, y, absdiff, levels=14, cmap="RdBu_r")

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
ax.plot(x, y, 'ko', ms=3)
ax.set_title('abs(difference) between every grid')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


plt.subplots_adjust(hspace=0.5)
plt.show()

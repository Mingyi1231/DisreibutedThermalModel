#-- copmute chiplets' temperature based on characterized self- and mutual- thermal resistance

from system import System_25D
import numpy as np 
import os
import sys
import csv
import math
import time
import config
import configparser
import util.fill_space
import subprocess
from scipy import interpolate
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool 

#-- remove blank lines in a file and dump to a new file
def rm_blank_line(filein, fileout):
    with open(filein, 'r') as r, open(fileout, 'w') as o:
        for line in r:
            #strip() function
            if line.strip():
                o.write(line)
    r.close()
    o.close()
    return


#-- read csv file
#-- format: entries separated by tabs
def readcsvfile(file):
    entries = list(csv.reader(open(file, 'r'), delimiter='\t'))
    return entries

def ReadCSVfile(filename):
    array = np.genfromtxt(filename)
    return array
# function to get unique values
def unique(list1):

    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


#-- read floorplan file
#-- return a list of chiplets info (name, coordinate, size)
#-- flp file format: unitname\tdx\tdy\tx0\ty0
def readflpfile(flpfile):
    global int_width, int_height    #-- interposer's width and height

    blocks = list(csv.reader(open(flpfile, 'r'), delimiter='\t'))
    max_x = 0
    max_y = 0
    i = 0
    chiplets_name = []
    chiplets_left = np.array([])
    chiplets_width = np.array([])
    chiplets_bottom = np.array([])
    chiplets_height = np.array([])
    for e in blocks:
        #-- skip blank lines
        if len(blocks[i]) == 0:
            i += 1
            continue
        #-- skip comment lines
        if blocks[i][0][0] == '#':
            i += 1
            continue
        chiplets_name += [blocks[i][0]]
        left = float(blocks[i][3])
        chiplets_left = np.append(chiplets_left, left)
        width = float(blocks[i][1])
        chiplets_width = np.append(chiplets_width, width)
        bottom = float(blocks[i][4])
        chiplets_bottom = np.append(chiplets_bottom, bottom)
        height = float(blocks[i][2])
        chiplets_height = np.append(chiplets_height, height)

        right = left + width
        top = bottom + height
        if max_x < right:
            max_x = right
        if max_y < top:
            max_y = top
        i += 1

    int_width = max_x
    int_height = max_y
    return chiplets_name, chiplets_left, chiplets_width, chiplets_bottom, chiplets_height





#--[TODO] parallel processing for speed


def gen_interpolate(cidx, chiplets_count, intp_size, chiplets_left, chiplets_bottom, chiplets_width, chiplets_height, chiplets_power, loc1distributedR, loc2distributedR, loc3distributedR, loc4distributedR):

    intp_width = intp_size
    intp_height = intp_size
    
    #-- intp_size in millimeter[mm], chiplet left, bottom, width, height in meter[m], distributed thermal resistance 2D table,(Dx,Dy) in millimmeter[mm]
    
#    INTP_X_GRID_COUNT = 64
#    INTP_Y_GRID_COUNT = 64
    INTP_X_GRID_COUNT = 128
    INTP_Y_GRID_COUNT = 128

    KTOC = 273.15
    TAMB = 45.0
    assert(len(chiplets_left) == len(chiplets_bottom))
    assert(len(chiplets_width) == len(chiplets_height))
    assert(len(chiplets_width) == len(chiplets_power))

    #-- 2D distributed thermal resistance data processing
    x1 = loc1distributedR[:,0]
    x2 = loc2distributedR[:,0]
    x3 = loc3distributedR[:,0]
    x4 = loc4distributedR[:,0]

    y1 = loc1distributedR[:,1]
    y2 = loc2distributedR[:,1]
    y3 = loc3distributedR[:,1]
    y4 = loc4distributedR[:,1]
    
    z1 = loc1distributedR[:,2]
    z2 = loc2distributedR[:,2]
    z3 = loc3distributedR[:,2]
    z4 = loc4distributedR[:,2]

    xx1 = unique(x1)
    xx2 = unique(x2)
    xx3 = unique(x3)
    xx4 = unique(x4)

    yy1 = unique(y1)
    yy2 = unique(y2)
    yy3 = unique(y3)
    yy4 = unique(y4)

    step_count1 = len(xx1)
    step_count2 = len(xx2)
    step_count3 = len(xx3)
    step_count4 = len(xx4)

    zz1 = np.reshape(z1, (-1, step_count1))
    zz2 = np.reshape(z2, (-1, step_count2))
    zz3 = np.reshape(z3, (-1, step_count3))
    zz4 = np.reshape(z4, (-1, step_count4))

    f2d1 = interpolate.interp2d(xx1, yy1, zz1, kind = 'cubic')
    f2d2 = interpolate.interp2d(xx2, yy2, zz2, kind = 'cubic')
    f2d3 = interpolate.interp2d(xx3, yy3, zz3, kind = 'cubic')
    f2d4 = interpolate.interp2d(xx4, yy4, zz4, kind = 'cubic')
    
    return f2d1, f2d2, f2d3, f2d4


#--[TODO]how to input the width, height, left, bottom, indx(chiplet serial number) parameter
def bigloop(location):
    gridR = [0,0]
    ii, jj = location[0], location[1]
    #--[TODO]intp_width, intp_height, cxx, cyy generating inside the bigloop
    x=jj*intp_width/INTP_X_GRID_COUNT
    y = intp_height*( 1.0 - ii/INTP_Y_GRID_COUNT)
    dxx = x - cxx
    dyy = y - cyy
    if dxx >= 0:
        if dyy >= 0:
            rm_grid = f2d1(dxx, dyy)
        else:
            rm_grid = f2d3(dxx, dyy)
    else:
        if dyy >= 0:
            rm_grid = f2d2(dxx, dyy)
        else:
            rm_grid = f2d4(dxx, dyy)
    
    #print("type_rm_grid",type(rm_grid))

    index = ii*INTP_Y_GRID_COUNT+jj
    gridR[0], gridR[1] = index, rm_grid[0]
    #print("type_rm_grid[0]",type(rm_grid[0]))
    #print("gridR",gridR)
    #print("gridR_type", type(gridR))
    return gridR


#--take the HotSpot translate mode for translate chiplet temperature from grid to block
def grid2chiplet_temp(path, gridfile, intp_size, chiplet_width, chiplet_height, chiplet_left, chiplet_bottom):
    
#    INTP_X_GRID_COUNT = 64
#    INTP_Y_GRID_COUNT = 64    
    INTP_X_GRID_COUNT = 128
    INTP_Y_GRID_COUNT = 128   

    KTOC = 273.15
    TAMB = 45.0
    intp_width = intp_size
    intp_height = intp_size
    chiplet_width = 1e3*chiplet_width
    chiplet_height = 1e3*chiplet_height
    chiplet_left = 1e3*chiplet_left
    chiplet_bottom = 1e3*chiplet_bottom
    LeftDownX = chiplet_left
    LeftDownY = chiplet_bottom
    RightUpX = chiplet_left + chiplet_width
    RightUpY = chiplet_bottom + chiplet_height

    res_array = ReadCSVfile(path + gridfile)
    npts = len(res_array)
    grid_temp = res_array[:,1] - 273.15
    grid_xunit = intp_width/INTP_X_GRID_COUNT
    grid_yunit = intp_height/INTP_Y_GRID_COUNT

    chiplet_maxT = 0
    j1 = math.ceil(LeftDownX*INTP_X_GRID_COUNT/intp_width)
    i1 = math.ceil((intp_height - LeftDownY)*INTP_Y_GRID_COUNT/intp_height)
    j2 = math.ceil(RightUpX*INTP_X_GRID_COUNT/intp_width)
    i2 = math.ceil((intp_height - RightUpY)*INTP_Y_GRID_COUNT/intp_height)
    ci1 = (i1+i2)//2
    cj1 = (j1+j2)//2
    
    ci2 = ci1 if ((i2-i1)%2) else ci1-1
    cj2 = cj1 if ((j2-j1)%2) else cj1-1

    ci1cj1 = ci1 * INTP_Y_GRID_COUNT + cj1
    ci2cj1 = ci2 * INTP_Y_GRID_COUNT + cj1
    ci1cj2 = ci1 * INTP_Y_GRID_COUNT + cj2
    ci2cj2 = ci2 * INTP_Y_GRID_COUNT + cj2
    
    chiplet_T = (grid_temp[ci1cj1]+grid_temp[ci2cj1]+grid_temp[ci1cj2]+grid_temp[ci2cj2])/4
    
    return chiplet_T

def clean_hotspot(path, stepfilename):
    os.system('rm ' + path + stepfilename + '{*.flp,*.lcf,*.ptrace,*.steady}')

def unique_WH(chiplets_widths, chiplets_heights):
    d = defaultdict(int)
    seen = set()
    idx = 0
    unique_widths, unique_heights = [], []
    for w, h in zip(chiplets_widths, chiplets_heights):
        if (w, h) not in seen:
            d[(w, h)] = idx
            idx += 1
            seen.add((w, h))
            unique_widths.append(w)
            unique_heights.append(h)
    return d, unique_widths, unique_heights  #-- d is a dictionary: {key: value} = {(w, h):group index}

#--bigloop should not be called in  function compute_tmax, but main function



if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 compute_temp.py <config_file>")
        print("<config_file>: TAP2.5D config file, e.g. configs/sys_micro150.cfg")
        sys.exit(1)

    cfgfile = sys.argv[1]
    sys_name = os.path.splitext(cfgfile)[0]
    #step_file = sys.argv[2]


    #-- global consts
#    INTP_X_GRID_COUNT = 64
#    INTP_Y_GRID_COUNT = 64
    INTP_X_GRID_COUNT = 128
    INTP_Y_GRID_COUNT = 128
    KTOC = 273.15
    TAMB = 45.0

    #-- this is the real temperature computation which will be called from RL
    tstart = time.time()
    chiplets_temp = np.array([])
    tmax = 0.0

    #Tmax, chiplets_temp = compute_tmax(cfgfile)








    
    #-- read in TAP2.5D config file, which contains chiplets info
    insys = config.read_config(cfgfile)

    chiplets_count = insys.chiplet_count
    intp_size = insys.intp_size
    global intp_width, intp_height
    intp_width = intp_size
    intp_height = intp_size
    chiplets_width = 1.0e-3*np.array(insys.width)    # [mm] -> [m]
    chiplets_height = 1.0e-3*np.array(insys.height)    # [mm] -> [m]
    assert(0 < len(insys.x))
    chiplets_x = 1.0e-3*np.array(insys.x)    # [mm] -> [m]
    assert(0 < len(insys.y))
    assert(len(insys.x) == len(insys.y))
    chiplets_y = 1.0e-3*np.array(insys.y)    # [mm] -> [m]
    chiplets_power = np.array(insys.power)
    sys_path = insys.path    # result output file location
    
    insys.gen_flp("table_model")

    chiplets_left = np.array([])
    chiplets_bottom = np.array([])
    #chiplets_width = np.array([])
    #chiplets_height = np.array([])
    for i in range(0, chiplets_count, 1):
        chiplets_left = np.append(chiplets_left, chiplets_x[i]-0.5*chiplets_width[i])
        chiplets_bottom = np.append(chiplets_bottom, chiplets_y[i]-0.5*chiplets_height[i])


    #-- this is the grid.steady temperature computation 
    
    compute_time_start = time.time()
    loc1distributedR_list = []
    loc2distributedR_list = []
    loc3distributedR_list = []
    loc4distributedR_list = []
    WH_dict, unique_width, unique_height = unique_WH(chiplets_width, chiplets_height)
    chiplet_groupnum = len(WH_dict)
    for i in range(0, chiplet_groupnum, 1):
        chiplet_name = "Chiplet" + str(i)
        loc1distributedR = np.loadtxt(sys_path+chiplet_name+"loc1.distributedR", delimiter='\t')
        loc2distributedR = np.loadtxt(sys_path+chiplet_name+"loc2.distributedR", delimiter='\t')
        loc3distributedR = np.loadtxt(sys_path+chiplet_name+"loc3.distributedR", delimiter='\t')
        loc4distributedR = np.loadtxt(sys_path+chiplet_name+"loc4.distributedR", delimiter='\t')
        
        loc1distributedR_list.append(loc1distributedR)
        loc2distributedR_list.append(loc2distributedR)
        loc3distributedR_list.append(loc3distributedR)
        loc4distributedR_list.append(loc4distributedR)

    grid_temp = np.array([])
#    chiplets_grid = np.array([0]*(INTP_X_GRID_COUNT*INTP_Y_GRID_COUNT))
    global cxx, cyy
    location = [[i,j] for i in range(INTP_X_GRID_COUNT) for j in range(INTP_Y_GRID_COUNT)]

    chiplets_grid = np.zeros(INTP_X_GRID_COUNT*INTP_Y_GRID_COUNT)

    for i in range(chiplets_count):
        chiplet_name = "Chiplet_" + str(i)
        wh = list(zip(chiplets_width, chiplets_height))
        listnum = WH_dict[(wh[i])]
        
        cxx = 1.0e3*chiplets_x[i]
        cyy = 1.0e3*chiplets_y[i]
        

        #chiplet_grid = compute_grid_temp(i, chiplets_count, intp_size, chiplets_left, chiplets_bottom, chiplets_width, chiplets_height, chiplets_power, loc1distributedR_list[listnum], loc2distributedR_list[listnum], loc3distributedR_list[listnum], loc4distributedR_list[listnum])
        f2d1, f2d2, f2d3, f2d4 = gen_interpolate(i, chiplets_count, intp_size, chiplets_left, chiplets_bottom, chiplets_width, chiplets_height, chiplets_power, loc1distributedR_list[listnum], loc2distributedR_list[listnum], loc3distributedR_list[listnum], loc4distributedR_list[listnum])

        pool = Pool()
        gridR = pool.map(bigloop, location)
        pool.close()
        pool.join()

#        print("final gridR type=", type(gridR))
        #--[TODO]compute the temperature from final gridR
        chiplet_grid = np.zeros(INTP_X_GRID_COUNT*INTP_Y_GRID_COUNT)
        for every_gridR in gridR:
            index, rm_grid = every_gridR[0], every_gridR[1]
            chiplet_grid[index] = rm_grid*chiplets_power[i]+TAMB+KTOC  
            

        #-- save to csv file for one chiplet and its power grid.steady file
        np.savetxt(sys_path+chiplet_name+".grid.steady", chiplet_grid, fmt='%.5f', delimiter='\t')
        
        #-- each chiplet with its power has a grid.steady, a superposition of chiplets to get the entire interposer temperature are required.
        grid_num = len(chiplet_grid)
        for i in range(grid_num):
            chiplets_grid[i] += chiplet_grid[i]

    for j in range(INTP_X_GRID_COUNT*INTP_Y_GRID_COUNT):
        chiplets_grid[j] = chiplets_grid[j] - (chiplets_count-1)*(KTOC+TAMB)
    
    #-- save to csv file for all chiplets grid.steady file
    grid_seq = np.array([])
    for j in range(INTP_X_GRID_COUNT*INTP_X_GRID_COUNT):
        grid_seq = np.append(grid_seq,j)
    grid_temp = np.vstack((grid_seq, chiplets_grid)).T
    np.savetxt(sys_path+"table_model.grid.steady", grid_temp, fmt='%.5f', delimiter='\t')

    compute_time_end = time.time()
    compute_time = compute_time_end - compute_time_start 
    print("compute_time=",compute_time)

    chiplets_temp = np.array([])
    tmax = 0
    for i in range(chiplets_count):
        chiplet_name = "Chiplet_"+str(i)
        gridfile = "table_model.grid.steady"
        chiplet_temp = grid2chiplet_temp(sys_path, gridfile, intp_size, chiplets_width[i], chiplets_height[i], chiplets_left[i], chiplets_bottom[i])
        chiplets_temp = np.append(chiplets_temp, chiplet_temp)
#        print(chiplet_name+": temperature",chiplets_temp[i])
        tmax = chiplet_temp if chiplet_temp > tmax else tmax













#    tend = time.time()
    chiplets_count = len(chiplets_temp)
    #-- print result
    for i in range(0, chiplets_count, 1):
        print("Chiplet: ",  "Chiplet_"+str(i), " Temp: ", chiplets_temp[i])
    print("Tmax:", tmax)



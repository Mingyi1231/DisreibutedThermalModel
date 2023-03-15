#-- copmute self and mutual thermal resistance by calling HotSpot

from system import System_25D
import os
import sys
import numpy as np 
import csv
import math
import config
import configparser
import util.fill_space
import subprocess
from collections import defaultdict


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


#-- read floorplan file
#-- return a list of chiplets info (name, coordinate, size)
#-- flp file format: unitname\tdx\tdy\tx0\ty0
def readflpfile(flpfile):
    global intp_width, intp_height    #-- interposer's width and height

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

    intp_width = max_x
    intp_height = max_y
    return chiplets_name, chiplets_left, chiplets_width, chiplets_bottom, chiplets_height


#-- write chiplet config file
def write_config(configname, syspath, intp_size, width, height, x, y):
    with open (configname, 'w') as Cfg:
        Cfg.write("[general]\n")
        Cfg.write("path = " + syspath + "\n")
        Cfg.write("placer_granularity = 1\n")
        Cfg.write("initial_placement = given\n")
        Cfg.write("decay = 0.8\n")
        Cfg.write("\n")

        Cfg.write("[interposer]\n")
        Cfg.write("# we will support passive, active, (photonic), and EMIB options.\n")
        Cfg.write("intp_type = passive\n")
        Cfg.write("intp_size = " + str(intp_size) + '\n')
        Cfg.write("link_type = nppl\n")
        Cfg.write("\n")
        Cfg.write("[chiplets]\n")
        Cfg.write("chiplet_count = 1\n")
        Cfg.write("widths = " + '\t' + str(width) + '\n')
        Cfg.write("heights = " + '\t' + str(height) + '\n')
        Cfg.write("powers = " + '\t' + str(100) + '\n')
        Cfg.write("x = " + str(x) + '\n')
        Cfg.write("y = " + str(y) + '\n')
        Cfg.write("\n")
        Cfg.write("connections = " + str(0) +'\n')
        return


#-- characterize self thermal resistance of a chiplet
def char_self_r(path, sys_name, chiplet_name, intp_size, chiplet_width, chiplet_height, chiplet_power):
    #global intp_width, intp_height    #-- [mm]
    INTP_X_GRID_COUNT = 64
    INTP_Y_GRID_COUNT = 64
    KTOC = 273.15
    TAMB = 45.0
    CHAR_STEP = 0
    CHIPLET_CHAR_POWER = 100.0
    intp_width = intp_size
    intp_height = intp_size

    config_name = sys_name + '_' + chiplet_name + '.cfg'

    x_start = 0.5*chiplet_width + 0.1
    x_end = 0.5*intp_width + 0.1
    x_step = 0.99*(x_end - x_start)
    y_start = 0.5*chiplet_height + 0.1
    y_end = 0.5*intp_height + 0.1
    y_step = 0.99*(y_end - y_start)

    x_arr = np.array([])
    y_arr = np.array([])
    rself_arr = np.array([])
    for y in np.arange(y_start, y_end, y_step):
        for x in np.arange(x_start, x_end, x_step):
            x_arr = np.append(x_arr, x)
            y_arr = np.append(y_arr, y)

            write_config(config_name, path, intp_size, chiplet_width, chiplet_height, x, y)
            system = config.read_config(config_name)
            system.gen_flp('step_'+str(CHAR_STEP))
            system.gen_ptrace('step_'+str(CHAR_STEP))
            print("#info# Characterizing R_self for chiplet group", chiplet_name, "at [",x,",", y, "] [mm]")
            chiplet_temp = system.run_hotspot('step_'+str(CHAR_STEP))
            chiplet_temp_diff = chiplet_temp - TAMB
            rself = chiplet_temp_diff / chiplet_power
            rself_arr = np.append(rself_arr, rself)


    #-- following are to be used in table loop-up
    x1 = np.arange(x_start, x_end, x_step)
    y1 = np.arange(y_start, y_end, y_step)
    rself_new = np.reshape(rself_arr, (-1, len(y1)))

    #-- convert [mm] to [m]
    x_arr *= 1.0e-3
    y_arr *= 1.0e-3
    xyr_arr = np.vstack((x_arr, y_arr, rself_arr)).T

    #-- save self R to file
    np.savetxt(path+chiplet_name+".rself", xyr_arr, fmt='%e', delimiter='\t')
    print("#info# R_self characterization done for chiplet group:", chiplet_name, ". Saved to file:", path+chiplet_name+".rself")
    
    return    # end of char_self_r()

def char_mutu_r(path, sys_name, chiplet_name, intp_size, chiplet_width, chiplet_height, chiplet_power):
    #global intp_width, intp_height
    INTP_X_GRID_COUNT = 64
    INTP_Y_GRID_COUNT = 64
    KTOC = 273.15
    TAMB = 45.0
    CHAR_STEP = 0
    CHIPLET_CHAR_POWER = 100.0
    intp_width = intp_size
    intp_height = intp_size

    config_name = sys_name + '_' + chiplet_name + '.cfg'

    xx = 0.5*chiplet_width + 0.1
    yy = 0.5*chiplet_height + 0.1
    write_config(config_name, path, intp_size, chiplet_width, chiplet_height, xx, yy)
    system = config.read_config(config_name)
    system.gen_flp('step_'+str(CHAR_STEP))
    system.gen_ptrace('step_'+str(CHAR_STEP))
    print("#info# Characterizing R_mutu for chiplet group", chiplet_name, "at [",xx,",", yy, "] [mm]")
    chiplet_temp = system.run_hotspot('step_'+str(CHAR_STEP))

    #-- find hotspot's interposer grid list
    #-- remove the blank lines in grid.steady file
    rm_blank_line(path+"step_"+str(CHAR_STEP)+".grid.steady", path+"step_"+str(CHAR_STEP)+".grid.steady1")
    grid_list = readcsvfile(path+"step_"+str(CHAR_STEP)+".grid.steady1")
    grid_columns = list(zip(*grid_list))

    #-- loop through hotspot grids
    x_grid_count = INTP_X_GRID_COUNT
    y_grid_count = INTP_Y_GRID_COUNT
#    dist = np.array([])
    rmut = np.array([])
    Dx = np.array([])
    Dy = np.array([])

    c_center_x = xx
    c_center_y = yy
#    c_center_x = 1.0e-3*xx
#    c_center_y = 1.0e-3*yy
    k = 0
    
    for i in range(y_grid_count):
        for j in range(x_grid_count):
            x = j*intp_width/x_grid_count
            y = intp_height*(1.0 - i/y_grid_count)
#            x = 1.0e-3*j*intp_width/x_grid_count
#            y = 1.0e-3*intp_height*(1.0 - i/y_grid_count)
#            dx = c_center_x - x
#            dy = c_center_y - y
            dx = x - c_center_x
            dy = y - c_center_y
            Dx = np.append(Dx, dx)
            Dy = np.append(Dy, dy)
#            dist = np.append(dist, math.sqrt(dx*dx+dy*dy))
            dt = float(grid_columns[1][k]) - KTOC - TAMB
            rmut = np.append(rmut, dt/chiplet_power)
            assert(0.0 <= dt)
            k += 1
            
    assert(k == x_grid_count*y_grid_count)

    #-- sort by distance
#    assert(dist.ndim == rmut.ndim)
    assert(Dx.ndim == rmut.ndim)
    assert(Dx.shape == rmut.shape)
    #dr = np.vstack((dist, rmut)).T
#    dr = np.vstack((rmut, dist)).T
    newdr = np.vstack((Dx, Dy, rmut)).T
#    dr_sorted = dr[np.lexsort(dr.T)]
#    dr_sorted[:, [1, 0]] = dr_sorted[:, [0, 1]]    #-- swap the 2 columns

    #-- save to csv file
#    np.savetxt(path+chiplet_name+".rmutu", dr_sorted, fmt='%e', delimiter='\t')
    np.savetxt(path+chiplet_name+".rmutu", newdr, fmt='%e', delimiter='\t')
    print("#info# R_mutual characterization done for chiplet group:", chiplet_name, "Saved to file:", path+chiplet_name+".rmutu")
    

    return    #-- end of char_mutu_r()


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

#def unique_WH(chiplets_widths, chiplets_heights):
#    d = defaultdict(int)
#    d1 = defaultdict(int)
#    seen = set()
#    seen1 = set()
#    idx = 0
#    unique_widths, unique_heights = [ ], [ ]
#    rotate_widths, rotate_heights = [ ], [ ]
#
#    for w, h in zip(chiplets_widths, chiplets_heights):
#        if(w, h) not in seen:
#            d[(w,h)] = idx
#            idx+=1
#            seen.add ((w, h))
#            unique_widths.append(w)
#            unique_heights.append(h)
#    indx = 0
#    for w, h in seen:
#        if(h,w) not in seen1:
#            d1[(w,h)] = indx
#            indx += 1
#            seen1.add((w,h))
#            rotate_widths.append(w)
#            rotate_heights.append(h)
#
#    return d1, rotate_widths, rotate_heights

if __name__ == "__main__":


    if len(sys.argv) != 2:
        print("Usage: python3 char_thermal_r.py <config_file>")
        print("<config_file>: TAP2.5D config file, e.g. configs/sys_micro150.cfg")
        sys.exit(1)

    #-- assume TAP2.5D config file is located in a directory
    cfgfile = sys.argv[1]
    sys_name = os.path.splitext(cfgfile)[0]
 
    #-- global consts
    INTP_X_GRID_COUNT = 64
    INTP_Y_GRID_COUNT = 64
    KTOC = 273.15
    TAMB = 45.0
    CHAR_STEP = 0
    CHIPLET_CHAR_POWER = 100.0

    #-- read in TAP2.5D config file, which contains chiplets info
    insys = config.read_config(cfgfile)

    chiplet_count = insys.chiplet_count
    intp_size = insys.intp_size
    intp_width = intp_size
    intp_height = intp_size
    chiplets_width = insys.width
    chiplets_height = insys.height
    chiplets_power = insys.power    # not used in characterization
    sys_path = insys.path

    WH_dict, unique_widths, unique_heights = unique_WH(chiplets_width, chiplets_height)
    chiplet_groupnum = len(WH_dict)
    #-- loop through each chiplet in the system
#    for i in range(0, chiplet_count, 1):
#        chiplet_name = "Chiplet_" + str(i)
#        char_self_r(sys_path, sys_name, chiplet_name, intp_size, chiplets_width[i], chiplets_height[i], CHIPLET_CHAR_POWER)
#        char_mutu_r(sys_path, sys_name, chiplet_name, intp_size, chiplets_width[i], chiplets_height[i], CHIPLET_CHAR_POWER)

    #-- create a list of unique sizes of chiplets
    for i in range(0, chiplet_groupnum, 1):
        chiplet_name = "Chiplet" + str(i)
        char_self_r(sys_path, sys_name, chiplet_name, intp_size, unique_widths[i], unique_heights[i], CHIPLET_CHAR_POWER)
        char_mutu_r(sys_path, sys_name, chiplet_name, intp_size, unique_widths[i], unique_heights[i], CHIPLET_CHAR_POWER)
    
#    for i, (w, h) in enumerate(zip(chiplets_width, chiplets_height)):
#        chiplet_name = "Chiplet_" + str(i)
#        index = WH_dict[(w, h)]
#        os.system('cp ' + sys_path + "Chiplet" + str(index) + ".rself" + ' ' + sys_path + chiplet_name + ".rself")
#        os.system('cp ' + sys_path + "Chiplet" + str(index) + ".rmutu" + ' ' + sys_path + chiplet_name + ".rmutu")



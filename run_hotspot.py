#-- read a TAP2.5D config file, generate HotSpot input files, and run HotSpot

from system import System_25D
import os
import sys
import csv
import time
import math
import numpy as np
import config
import configparser
import util.fill_space
import subprocess


def run_hotspot_1(path, filename):
    proc = subprocess.Popen(["./util/hotspot",
        "-c",path+"new_hotspot.config",
	"-f",path+filename+"L4_ChipLayer.flp",
	"-p",path+filename+".ptrace",
	"-steady_file",path+filename+".steady",
	"-grid_steady_file",path+filename+".grid.steady",
	"-model_type","grid",
	"-detailed_3D","on",
        #--test for more grid count
        "-grid_rows","128",
        "-grid_cols","128",
	"-grid_layer_file",path+filename+"layers.lcf"], 
    stdout=subprocess.PIPE, stderr = subprocess.PIPE)
    stdout, stderr = proc.communicate()
    outlist = stdout.split()
    return (max(list(map(float,outlist[3::2])))-273.15)

def clean_hotspot(path, filename):
    os.system('rm ' + path + filename + '{*.flp,*.lcf,*.ptrace,*.steady}')


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 run_hotspot.py <config_file>")
        print("<config_file>: TAP2.5D config file, e.g. configs/sys_micro150.cfg")
        sys.exit(1)


    RUN_STEP = 999

    tstart = time.time()
    system = config.read_config(sys.argv[1])
    path = system.path
    system.gen_flp('step_'+str(RUN_STEP))
    system.gen_ptrace('step_'+str(RUN_STEP))
    tmax = system.run_hotspot('step_'+str(RUN_STEP))

    #temp_new = run_hotspot("./output/micro150/", sys.argv[1])
    tend = time.time()

    #-- get all chiplets temp
    entries = list(csv.reader(open(path+'step_'+str(RUN_STEP)+".steady", 'r'), delimiter='\t'))
    cols = list(zip(*entries))
    cnames = cols[0]
    temps = cols[1]
    print("HotSpot result")
    i = 0
    for cn in cnames:
        if ('layer_4_Chiplet' in cn):
            print(cn, ":", float(temps[i])-273.15)
        i += 1

    print("max temp:", tmax)
    print("runtime:", tend-tstart)


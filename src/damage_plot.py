import rainflow
import numpy as np

from config import CONFIG

CONTROL_COLUMNS = CONFIG["data_module_kwargs"]["setup_kwargs"]["control_columns"]
TARGET_COLUMNS = CONFIG["data_module_kwargs"]["setup_kwargs"]["target_columns"]

def damage_index(signal):
    # Parameters for damage computation

    # Slope of the curve
    m=8

    # number of cycle for knee point
    Nk=2e6

    # Ultimate stress
    Rm = 865

    # [MPa] Endurance Limit
    sigaf=Rm/2  

    # Computation : Rainflow counting algorithm
    #ext, exttime = turningpoints(signal, dt)
    rf = rainflow.extract_cycles(signal) # range, mean, count, i_start, i_end
    rf = np.array([i for i in rf]).T
    rf[1] = 0.5* rf[1] # Matlab function is half as big for some reason

    siga = (rf[0]*Rm)/(Rm-rf[1])  #rf[2] # Mean stress correction (Goodman)
    rf[0] = siga
    damage_full = (rf[2]/Nk) * (siga/sigaf)**m
    damage = np.sum(damage_full)

    return damage, rf, damage_full

def find_type(in_str):
    file = in_str.split('/')[-1]
    type_ = file.split('_')[0]
    return type_

def convert_numeric_string(num):
    if num[-1].isnumeric():
        if num[0].isnumeric():
            return int(num[0]) + int(num[2]) / 10
        else:
            return int(num[-1])
    else:
        return 1.0

def find_fsim(in_str):
    folder = in_str.split('/')[-2]
    num = folder[-3:]
    return convert_numeric_string(num)

def find_type_and_fsim(in_str):
    # String magic to find the type (BEP, 2Slopes, Linear, Classic)
    # and Froude similarity from the path of the file
    if '/' in in_str:
        return find_type(in_str), find_fsim(in_str)
    else:
        return in_str.split('_')[1], convert_numeric_string(in_str[-3:])

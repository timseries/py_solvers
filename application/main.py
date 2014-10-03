#!/usr/bin/python -tt
"""
Python script to prompt the user with a number of applications. Then allow selecting the configuration.
"""
import os
import sys
from py_utils.parameter_struct import ParameterStruct
from py_utils.helpers import convertStr
from py_utils.section_factory import SectionFactory as sf 

from matplotlib import pyplot as plt
import time

def main():
    #configuration specification, absolute path
    seed_flag = -1
    if len(sys.argv)>1:
        ps_path=sys.argv[1]
        if len(sys.argv)>2:
            seed_flag = sys.argv[2]
    else:    
        ps_path='/home/tim/repos/py_solvers/applications/deconvolution/config/downsampled_cameraman_vbmm.ini'
    ps_params = ParameterStruct(ps_path)
    if seed_flag!=-1:
        #set the seed of sec_observe to the commandline arg
        ps_params.set_key_val_pairs('Observe1', ['seed'], [seed_flag])
    dict_in = {}
    sec_input = sf.create_section(ps_params,'Input1')
    sec_preprocess = sf.create_section(ps_params,'Preprocess1')
    sec_observe = sf.create_section(ps_params,'Observe1')
    so_solver = sf.create_section(ps_params,'Solver1')

    #read, observe, solve, report
    sec_input.read(dict_in)
    sec_preprocess.preprocess(dict_in)
    sec_observe.observe(dict_in)
    t0 = time.time()    
    so_solver.solve(dict_in)
    t1 = time.time()
    so_solver.results.display()
    so_solver.results.save()
    print "Solver finished in " + str(t1-t0) + " seconds"
if __name__ == "__main__":
    main()
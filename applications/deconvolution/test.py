#!/usr/bin/python -tt
"""
Python script to prompt the user with a number of applications. Then allow selecting the configuration.
"""
import os
from py_utils.parameter_struct import ParameterStruct
from py_utils.helpers import convertStr
from py_utils.section_factory import SectionFactory as sf 
from numpy import arange
from numpy.linalg import norm

def main():
    #configuration specification, absolute path
    ps_path='/home/tim/repos/py_solvers/applications/deconvolution/uniform_40db_bsnr_cameraman.ini'
    ps_params = ParameterStruct(ps_path)
    dict_in = {}
    sec_input = sf.create_section(ps_params,'Input1')
    sec_observe = sf.create_section(ps_params,'Observe1')
    so_solver = sf.create_section(ps_params,'Solver1')

    #read, observe, solve, report
    sec_input.read(dict_in)
    W = sec_observe.W.ls_operators[0]
    for n in arange(20):
        w_n = W * dict_in['x']
        x_n = ~W * w_n
        print norm(x_n.flatten() - dict_in['x'].flatten(),ord=2)
    
if __name__ == "__main__":
    main()
#! /home/zelda/tr331/ENVNOSITE/bin/python
#$ -S /home/zelda/tr331/ENVNOSITE/bin/python
#comment!/usr/bin/python -tt
"""
Python script to prompt the user with a number of applications. Then allow selecting the configuration.
"""
import os
from py_utils.parameter_struct import ParameterStruct
from py_utils.helpers import convertStr
from py_utils.section_factory import SectionFactory as sf 
from libtiff import TIFFimage
import Image
import numpy as np
import itertools

def main():
    #configuration specification, absolute path
    ps_path='~/repos/py_solvers/applications/deconvolution_challenge/p3.ini'
    ps_params = ParameterStruct(ps_path)
    dict_in = {}
    sec_input = sf.create_section(ps_params,'Input1')
    sec_observe = sf.create_section(ps_params,'Observe1')

    #read, preprocess, observe, solve, report
    sec_input.read(dict_in)
    sec_observe.observe(dict_in)
    so_solver = sf.create_section(ps_params,'Solver1')
    
    #works for p0,p1,p2
    # nustart_factors=(1,)
    # nustop_factors=(.7,)


    #case p3
    nustart_factors=(1,)#1.1,1.2,1.3,1.4)
    nustop_factors=(.2,)

    #alpha sweep
    # nustart_factors=(1,.9,.8,.7,.6) #really {1.5,3,4.5}*sigma_g
    # nustop_factors=(.5,.2,.1,.01,.01) #really {1,.7,.5}*sigma_g
    # nustop_factors=(.5) #really {1,.7,.5}*sigma_g
    
    ls_nu_factors=[nustart_factors,nustop_factors]
    nu_start = so_solver.get_val('nustart', True)
    nu_stop = so_solver.get_val('nustop', True)
    epsilon_start = so_solver.get_val('epsilonstart', True)
    epsilon_stop = so_solver.get_val('epsilonstop', True)
    alpha=so_solver.alpha
    ls_alpha_keys=['alpha']
    ls_nu_keys=['nustart','nustop']
    ls_epsilon_keys=['epsilonstart','epsilonstop']
    for nu_multiplier in itertools.product(*ls_nu_factors):
        #nu sweep
        ls_nu_vals=[nu_multiplier[0]*nu_start,nu_multiplier[1]*nu_stop]
        ps_params.set_key_val_pairs('Solver1',ls_nu_keys,ls_nu_vals)

        #epsilon sweep
        # ls_epsilon_vals=[nu_multiplier[0]*epsilon_start,nu_multiplier[1]*epsilon_stop]
        # ps_params.set_key_val_pairs('Solver1',ls_epsilon_keys,ls_epsilon_vals)

        #alpha sweep
        # ls_alpha_vals=[1.1*alpha]
        # ls_alpha_vals[0][0]=1.0
        # ps_params.set_key_val_pairs('Solver1',ls_alpha_keys,ls_alpha_vals)

        #create the solver and solve
        so_solver = sf.create_section(ps_params,'Solver1')
        so_solver.solve(dict_in)
        so_solver.results.save()
    
if __name__ == "__main__":
    main()

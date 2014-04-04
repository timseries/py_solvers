#commented !/usr/bin/python -tt
"""
Python script to prompt the user with a number of applications. Then allow selecting the configuration.
"""
import os
from py_utils.parameter_struct import ParameterStruct
from py_utils.helpers import convertStr
from py_utils.section_factory import SectionFactory as sf 

def main():
    #configuration specification, absolute path
    # ps_path='~/repos/py_solvers/applications/classification/cell_params.ini'
    # ps_path='/home/zelda/tr331/repos/py_solvers/applications/texture_classification/texture_params.ini'
    ps_path='~/repos/py_solvers/applications/classification/galaxy_classification.ini'
    ps_params = ParameterStruct(ps_path)
    dict_in = {}
    sec_input = sf.create_section(ps_params,'Input1')
    #sec_preprocess = sf.create_section(ps_params,'Preprocess1') implement  later
    sec_observe = sf.create_section(ps_params,'Observe1')
    so_solver = sf.create_section(ps_params,'Solver1')

    #read, observe, solve, report
    sec_input.read(dict_in)
    sec_observe.observe(dict_in)
    so_solver.solve(dict_in)
    so_solver.results.display()
    so_solver.results.save()

if __name__ == "__main__":
    main()
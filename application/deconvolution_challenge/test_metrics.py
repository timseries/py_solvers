#!/usr/bin/python -tt
import os
from py_utils.parameter_struct import ParameterStruct
from py_utils.helpers import convertStr
from py_utils.section_factory import SectionFactory as sf 
from numpy.fft import fftn, ifftn
from py_utils.signal_utilities.sig_utils import crop

def main():
    ps_path='/home/tim/repos/py_solvers/applications/deconvolution_challenge/test.ini'
    ps_params = ParameterStruct(ps_path)
    dict_in = {}
    sec_input = sf.create_section(ps_params,'Input1')
    sec_input.read(dict_in)
    #'x' now has the reconstruction, so move it to 'x_n'
    dict_in['x_n'] = dict_in['x']
    dict_in['x_n_f'] = fftn(dict_in['x_n'])
    sec_input2 = sf.create_section(ps_params,'Input3')
    sec_input2.read(dict_in)
    sec_observe = sf.create_section(ps_params,'Observe1')
    sec_observe.observe(dict_in)
    #now remove the extra padding from x
    results = sf.create_section(ps_params,'Results1')
    results.update(dict_in)
    
if __name__ == "__main__":
    main()

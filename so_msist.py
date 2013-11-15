#!/usr/bin/python -tt
import numpy as np
from py_utils.signal_utilities import ws as ws
from py_operators.operator import Operator
class MSIST(Section):
    """
    Solver which performs the MSIST algorithm, with several different variants.
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(Section,self).__init__(ps_parameters,str_section)
        self.str_solver_variant = self.get_val('solvervariant',False)
        self.str_nu_epsilon_method = self.get_val('nuepsilonmethod',False)
        self.str_alpha_method = self.get_val('alphaspectralradius',False)
        self.str_variance_method = self.get_val('variance_method',False)
        self.int_iterations = self.get_val('nitn',True)
        
    def solve(self,in_object):
        """
        Takes an input object (ground truth, forward model observation, metrics required)
        Returns a solution object based on the solver this object was instantiated with.
        """
        if self.str_solver_variant == 'solvereal'
            sol_object = self.solve_real(in_object)
        elif: self.str_solver_variant == 'solverealimag'
            sol_object = self.solve_real_imag(in_object)
        elif: self.str_solver_variant == 'solvegroupsparse' or \
            self.str_solver_variant == 'solvegroupsparsefast'
            sol_object = self.solvegroupsparse(in_object)
        else:             
            raise Exception("no such MSIST solver variant " + self.str_solver_variant \
                            + " supported")    
        return sol_object
    
    def solve_real(self,input):
        
        
    class Factory:
        def create(self,ps_parameters,str_section):
            return MSIST(ps_parameters,str_section)








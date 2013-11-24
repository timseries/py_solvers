#!/usr/bin/python -tt
from py_utils.section import Section

class Solver(Section):
    """
    Base class for defining custom solvers
    """
    
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for Operator.
        """       
        super(Solver,self).__init__(ps_parameters,str_section)
        self.int_iterations = self.get_val('nitn',True)
        self.results = Results(ps_parameters,self.get_val('results',False))    
        
    def solve(self,dict_in):
        self.results.clear() #start fresh

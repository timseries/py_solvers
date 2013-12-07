#!/usr/bin/python -tt
from py_utils.section import Section
from py_utils.section_factory import SectionFactory as sf

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
        self.results = sf.create_section(ps_parameters,self.get_val('results',False))    
        
    def solve(self):
        self.results.clear() #start fresh

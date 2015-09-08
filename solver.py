#!/usr/bin/python -tt
from py_utils.section import Section
from py_utils.section_factory import SectionFactory as sf

class Solver(Section):
    r"""Base class for defining custom solvers
    
    Attributes:
        int_iterations (:class:`float`): Maximum (total) number of iterations.
        results (:class:`py_utils.results.Results`): The results to store for a 
            single execution of the solve method.

    """
    
    def __init__(self,ps_parameters,str_section):
        """Class constructor for Operator.
        """
        
        super(Solver,self).__init__(ps_parameters,str_section)
        self.int_iterations = self.get_val('nitn',True)
        self.results = sf.create_section(ps_parameters, 
                                         self.get_val('results',False))
        
    def solve(self):
        r"""Execute solver.
        """
        self.results.clear() #start fresh by running this first in any solver

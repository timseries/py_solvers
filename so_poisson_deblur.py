#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj, sqrt, ones
from numpy.fft import fftn, ifftn
from numpy import abs as nabs, exp, maximum as nmax
from py_utils.signal_utilities.ws import WS
import py_utils.signal_utilities.sig_utils as su
from py_solvers.solver import Solver
from py_operators.operator import Operator
from py_operators.operator_comp import OperatorComp

class PoissonDeblur(Solver):
    """
    Solver which performs the Poisson Deblurring Algorithm (for widefield microscopy deconvolution)
    """
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(PoissonDeblur,self).__init__(ps_parameters,str_section)
        self.alpha = None
        self.H = OperatorComp(ps_parameters,self.get_val('modalities',False))
        self.W = OperatorComp(ps_parameters,self.get_val('transforms',False))
        self.W = self.W.ls_operators[0] #assume we just have one transform
        self.alpha = self.get_val('alpha',True)
        self.str_group_structure = self.get_val('grouptypes',False)

    def solve(self,dict_in):
        """
        Takes an input object (ground truth, forward model observation, metrics required)
        Returns a solution object based on the solver this object was instantiated with.
        """
        super(PoissonDeblur,self).solve()
        H = self.H
        W = self.W
        #input data 
        x_n = dict_in['x_0'].copy()
        b = dict_in['b']#background
        dict_in['x_n'] = x_n
        #group structure
        if self.str_group_structure == 'self':
            g_i = 1.0
        elif self.str_group_structure == 'complexself':
            g_i = 2.0
        elif self.str_group_structure == 'parentchildren':
            g_i = 2.0
        elif self.str_group_structure == 'parentchild':
            g_i = 2.0
        elif self.str_group_structure == '':
            g_i = 2.0
        else:
            raise Exception("no such group structure " + self.str_group_structure)
        #input parameters and initialization
        if self.alpha.__class__.__name__ != 'ndarray':
            self.alpha = su.spectral_radius(self.W,self.H,dict_in['x_0'].shape)
        alpha = self.alpha
        w_n = W * x_n
        S_n = WS(np.zeros(w_n.ary_scaling.shape),(w_n.one_subband(0)).tup_coeffs) #initialize the variance matrix as a ws object
        dict_in['w_n'] = w_n
        epsilon,nu = self.get_epsilon_nu()
        dict_in['nu_sq'] = nu**2
        dict_in['epsilon_sq'] = epsilon**2
        #begin iterations here
        self.results.update(dict_in)
        for n in np.arange(self.int_iterations):
            f_2 = H * x_n + b
            f_1 = f_2 + 3/8
            u = 2 * sqrt(f_1) - 1 / (4 * sqrt(f_2))
            u_sq = u**2
            u_sq_f_1 = np.dot(u_sq.flatten(),f_1.flatten())
            u_sq_f_2 = np.dot(u_sq.flatten(),f_2.flatten())

            f_resid = -u_sq/sqrt(u_sq_f_1) + sqrt(u_sq_f_2)/(4*f_2) \
              - u_sq/(8*f_2*sqrt(u_sq_f_2)) + 2*ones(u_sq_f_2.shape) - sqrt(2*f_2/f_1)*((f_1-f_2)/f_2**2) \
              - 1/(16*f_2**2)
            w_n = ~H * f_resid
            w_n = W * w_n
            for s in arange(1,w_n.int_subbands):
                #variance estimate
                S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                #update current solution
                w_n.set_subband(s, w_n.get_subband(s)/S_n.get_subband(s))
            x_n = ~W * w_n
            w_n = W * x_n #reprojection, to put our iterate in the range space, prevent drifting
            dict_in['x_n'] = x_n
            dict_in['w_n'] = w_n
            #update results
            self.results.update(dict_in)
        return dict_in

    def get_epsilon_nu(self):
        '''
        A method for generating the sequences of nu and epsilon using some continuation rule.
        '''
        str_method = self.get_val('nuepsilonmethod', False)
        decay = self.get_val('decay', True)
        epsilon_start = self.get_val('epsilonstart', True)
        epsilon_stop = self.get_val('epsilonstop', True)
        nu_start = self.get_val('nustart', True)
        nu_stop = self.get_val('nustop', True)
        if str_method == 'geometric':
            epsilon = np.asarray([nmax(epsilon_start * (decay ** i),epsilon_stop) \
                                  for i in arange(self.int_iterations)])
            nu = np.asarray([nmax(nu_start * (decay ** i),nu_stop) \
                                  for i in arange(self.int_iterations)])
        elif str_method == 'exponential':
            epsilon = np.asarray([epsilon_start * exp(-i / decay) + epsilon_stop \
                                  for i in arange(self.int_iterations)])
            nu = np.asarray([nu_start * exp(-i / decay) + nu_stop \
                                  for i in arange(self.int_iterations)])
        elif str_method == 'fixed':
            epsilon = epsilon_start
            nu = nu_start
        else:
            raise Exception('no such continuation parameter rule')
        return epsilon,nu
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return PoissonDeblur(ps_parameters,str_section)

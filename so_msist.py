#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj
from numpy.fft import fftn, ifftn
from numpy import abs as nabs
from py_utils.signal_utilities import ws as ws
import py_utils.signal_utilities.sig_utils as su
from py_solvers.solver import Solver
from py_operators.operator import Operator
from py_operators.operator_comp import OperatorComp

class MSIST(Solver):
    """
    Solver which performs the MSIST algorithm, with several different variants.
    """
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(MSIST,self).__init__(ps_parameters,str_section)
        self.str_solver_variant = self.get_val('solvervariant',False)
        self.str_nu_epsilon_method = self.get_val('nuepsilonmethod',False)
        self.str_variance_method = self.get_val('variance_method',False)
        self.alpha = None
        self.H = OperatorComp(ps_parameters,self.get_val('modalities',False))
        self.W = OperatorComp(ps_parameters,self.get_val('transforms',False))
        self.Hhat = self.H.get_spectrum
        self.Hhat_star = conj(self.Hhat)
        self.alpha = self.get_val('alpha',True)
        if self.alpha == 0: #alpha override wasn't supplied, so compute
            self.alpha = su.spectral_radius(self.T,self.H)
        self.str_group_structure = self.get_val('grouptypes',False)

    def solve(self,dict_in):
        """
        Takes an input object (ground truth, forward model observation, metrics required)
        Returns a solution object based on the solver this object was instantiated with.
        """
        super(MSIST,self).solve(ps_parameters,str_section)
        Hhat = self.Hhat
        W = self.W
        Hhat_star = self.Hhat_star
        #input data 
        y_hat = dict_in['y_hat']
        x_n = dict_in['x_0']
        ary_alpha = dict_in['alpha']
        nu = dict_in['nu']
        #group structure
        if self.str_group_structure == 'self':
            g_i = 1
        elif self.str_group_structure == 'complexself':
            g_i = 2
        elif self.str_group_structure == 'parentchildren':
            g_i = 2
        elif self.str_group_structure == 'parentchild':
            g_i = 2
        else:    
            raise Exception("no such group structure " + self.str_group_structure)
        #input parameters and initialization
        S_n = W * np.zeros(x_n.shape)
        w_n = W * x_n
        if self.str_solver_variant == 'solvereal': #msist
            epsilon = dict_in['epsilon']
        elif self.str_solver_variant == 'solvevbmm': #vbmm    
            p_a = dict_in['p_a']
            p_b_0 = dict_in['p_b_0']
            p_k = dict_in['p_k']
            p_theta = dict_in['p_theta']
            b_n = W * x_n
            for s in arange(ws_resid.int_subbands):
                b_n.set_subbands(s, p_b_0)
        else:
            raise Exception("no MSIST solver variant " + self.str_solver_variant)
        #begin iterations here
        for n in np.arange(self.int_iterations):
            w_resid = W * ifftn(conj(Hhat_star) * (y_hat - Hhat * fftn(x_n)))
            for s in arange(1,w_n.int_subbands):
                #variance estimate
                if self.str_solver_variant == 'solvereal': #msist
                    S_n.set_subbands(s,1 / (1 / g_i * nabs(w_n.get_subband(s))**2 + epsilon[n]**2))
                elif self.str_solver_variant == 'solvevbmm': #vbmm    
                    S_n.set_subband(s, (2 * p_a + g_i) / (2 * p_b + nabs(w_n.get_subband(s))**2))
                    b_n.set_subband(s, (p_k + p_a) / (p_theta + S_n.get_subband(s)))
                #update current solution
                w_n.set_subband(s, \
                  (ary_alpha[s] * w_n.get_subband(s)+w_resid.get_subband(s)) / \
                  (ary_alpha[s] + (nu[n]**2) * S_n.get_subbands(s)))
                #update results
                self.results.update(dict_in,x_n,w_n)
            x_n = ~W * w_n
            w_n = W * x_n #reprojection, to put our iterate in the range space, prevent drifting
        dict_in['x_n'] = x_n
        dict_in['w_n'] = w_n
        return dict_in
    
    class Factory:
        def create(self,ps_parameters,str_section):
            return MSIST(ps_parameters,str_section)

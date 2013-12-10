#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj
from numpy.fft import fftn, ifftn
from numpy import abs as nabs, exp, max as nmax
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
        self.str_variance_method = self.get_val('variance_method',False)
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
        super(MSIST,self).solve()
        H = self.H
        W = self.W
        print 'test'
        test = H * dict_in['x_0']
        print 'test over'
        #input data 
        y_hat = dict_in['yhat']
        x_n = dict_in['x_0'].copy()
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
        S_n = W * np.zeros(x_n.shape)
        w_n = W * x_n
        dict_in['w_n'] = w_n

        if self.str_solver_variant == 'solvereal': #msist
            epsilon,nu = self.get_epsilon_nu()
            dict_in['nu_sq'] = nu**2
            dict_in['epsilon_sq'] = epsilon**2
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
        self.results.update(dict_in)
        for n in np.arange(self.int_iterations):
            f_resid = ifftn(y_hat - H * x_n)
            w_resid = W * ifftn(~H * f_resid)
            for s in arange(1,w_n.int_subbands):
                #variance estimate
                if self.str_solver_variant == 'solvereal': #msist
                    S_n.set_subband(s,(1.0 / (1.0 / g_i * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                elif self.str_solver_variant == 'solvevbmm': #vbmm    
                    S_n.set_subband(s, (2.0 * p_a + g_i) / (2.0 * p_b + nabs(w_n.get_subband(s))**2))
                    b_n.set_subband(s, (p_k + p_a) / (p_theta + S_n.get_subband(s)))
                else:
                    raise Exception('no such solver variant')
                #update current solution
                w_n.set_subband(s, \
                  (alpha[s] * w_n.get_subband(s) + w_resid.get_subband(s)) / \
                  (alpha[s] + (nu[n]**2) * S_n.get_subband(s)))
            x_n = (~W) * w_n
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
            return MSIST(ps_parameters,str_section)

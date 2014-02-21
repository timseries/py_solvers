#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj, abs as nabs, exp, maximum as nmax
from numpy.fft import fftn, ifftn
from py_utils.signal_utilities.ws import WS
import py_utils.signal_utilities.sig_utils as su
from py_solvers.solver import Solver
from py_operators.operator import Operator
from py_operators.operator_comp import OperatorComp

import scipy.stats as ss
import pdb

class MSIST(Solver):
    """
    Solver which performs the MSIST algorithm, with several different variants.
    """
    def __init__(self,ps_params,str_section):
        """
        Class constructor for DTCWT
        """
        super(MSIST,self).__init__(ps_params,str_section)
        self.str_sparse_pen = self.get_val('sparsepenalty',False)
        self.alpha = None
        self.H = OperatorComp(ps_params,
                              self.get_val('modalities',False))
        if len(self.H.ls_ops)==1: #avoid slow 'eval' in OperatorComp
            self.H = self.H.ls_ops[0] 
        self.W = OperatorComp(ps_params,
                              self.get_val('transforms',False))
        if len(self.W.ls_ops)==1: #avoid slow 'eval' in OperatorComp
            self.W = self.W.ls_ops[0] 
        self.alpha = self.get_val('alpha',True)
        self.str_group_structure = self.get_val('grouptypes',False)
        print self.str_group_structure
    def solve(self,dict_in):
        """
        Takes an input object (ground truth, forward model observation, metrics required)
        Returns a solution object based on the solver this object was instantiated with.
        """
        super(MSIST,self).solve()
        H = self.H
        W = self.W
        #input data 
        y_hat = fftn(dict_in['y'])
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
        w_n = W * x_n
        S_n = WS(np.zeros(w_n.ary_lowpass.shape),(w_n.one_subband(0)).tup_coeffs) #initialize the variance matrix as a ws object
        dict_in['w_n'] = w_n

        if self.str_sparse_pen == 'l0rl2': #msist
            epsilon,nu = self.get_epsilon_nu()
            dict_in['nu_sq'] = nu**2
            dict_in['epsilon_sq'] = epsilon**2
        elif (self.str_sparse_pen == 'vbmm' or  #vbmm    
              self.str_sparse_pen == 'vbmm_hmt'):
            nu = self.get_val('nustart',True) * np.ones(self.int_iterations,)
            dict_in['nu_sq'] = nu**2
            p_a = self.get_val('p_a',True)
            p_b_0 = self.get_val('p_b_0',True)
            p_k = self.get_val('p_k',True)
            p_theta = self.get_val('p_theta',True)
            b_n = WS(np.zeros(w_n.ary_lowpass.shape),
                     (w_n.one_subband(0)).tup_coeffs)
            if self.str_sparse_pen == 'vbmm_hmt':
                ary_a = self.get_gamma_shapes(W * dict_in['x_0'])
            for s in arange(w_n.int_subbands):
                b_n.set_subband(s, p_b_0)
        else:
            raise Exception("no MSIST solver variant " + self.str_sparse_pen)
        #begin iterations here
        self.results.update(dict_in)
        for n in np.arange(self.int_iterations):
            H.output_fourier = 1
            f_resid = ifftn(y_hat - H * x_n)
            w_resid = W * ifftn(~H * f_resid)
            H.output_fourier = 0
            for s in xrange(w_n.int_subbands-1,-1,-1):
                #variance estimate
                if self.str_sparse_pen == 'l0rl2': #msist
                    S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                elif self.str_sparse_pen == 'vbmm': #vbmm    
                    if n == 0:
                        sigma_n = 0
                    else:
                        sigma_n = (1.0 / nu[n]**2 * alpha[s] + S_n.get_subband(s))**(-1)
                    S_n.set_subband(s, (g_i + 2.0 * p_a) / (nabs(w_n.get_subband(s))**2 + sigma_n + 2.0 * b_n.get_subband(s)))
                    b_n.set_subband(s, (p_k + p_a) / (S_n.get_subband(s) + p_theta))
                elif self.str_sparse_pen == 'vbmm_hmt': #vbmm    
                    if n == 0:
                        sigma_n = 0
                    else:
                        sigma_n = (1.0 / nu[n]**2 * alpha[s] + S_n.get_subband(s))**(-1)
                    if s < w_n.int_subbands - 6 and s > 0:    
                        s_parent = w_n.get_subband(s+6)
                        s_parent_us = np.zeros((2*s_parent.shape[0],2*s_parent.shape[1]))
                        s_parent_us[0::2,0::2]=s_parent
                        s_parent_us[0::2,1::2]=s_parent
                        s_parent_us[1::2,0::2]=s_parent
                        s_parent_us[1::2,1::2]=s_parent
                        adj_factor = .5
                        S_n.set_subband(s, (g_i + 2.0 * ary_a[s]) / 
                                        (nabs(w_n.get_subband(s))**2 + sigma_n + 
                                         2.0 * adj_factor * np.abs(s_parent_us)**2))
                    else: #no parents, so generate fixed-param gammas
                        S_n.set_subband(s, (g_i + 2.0 * p_a) / 
                                        (nabs(w_n.get_subband(s))**2 + sigma_n +
                                         2.0 * b_n.get_subband(s)))
                        b_n.set_subband(s, (p_k + p_a) / 
                                        (S_n.get_subband(s) + p_theta))
                else:
                    ValueError('no such solver variant')
                #update current solution
                w_n.set_subband(s, \
                  (alpha[s] * w_n.get_subband(s) + w_resid.get_subband(s)) / \
                  (alpha[s] + (nu[n]**2) * S_n.get_subband(s)))
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

    def get_gamma_shapes(self, ws_coeffs):
        """Use maximum likelihood to obtain the shape parameters 
        for each a gamma distribution.

        Return:
          ary_a (ndarray): a 1-d array of shape parameters
        """
        ary_a = np.zeros(ws_coeffs.int_subbands,)
        for s in xrange(ws_coeffs.int_subbands):
            subband = ws_coeffs.get_subband(s).flatten()
            subband_ri = np.concatenate((subband.real,subband.imag))
            fit_nu, fit_loc, fit_scale = ss.t.fit(subband_ri)
            ary_a[s] = fit_nu / 2
        return ary_a    
            
    class Factory:
        def create(self,ps_params,str_section):
            return MSIST(ps_params,str_section)

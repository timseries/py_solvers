#!/usr/bin/python -tt
import numpy as np
from numpy import arange, fftn, ifftn, conj
from py_utils.signal_utilities import ws as ws
import sig_utils as su
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
        self.str_variance_method = self.get_val('variance_method',False)
        self.int_iterations = self.get_val('nitn',True)
        self.alpha = None
        self.H = None #the modality (or composition of modalities)
        self.W = None #the transform (or composition of transforms)
        
    def solve(self,dict_in):
        """
        Takes an input object (ground truth, forward model observation, metrics required)
        Returns a solution object based on the solver this object was instantiated with.
        """
        #compute the variant-independent quantities here, if necessary
        if (dict_in['modality'] != self.H) and (dict_in['transform'] != self.W):
            self.H = dict_in['modality']
            self.W = dict_in['transform']
            dict_in['alpha'] = su.spectral_radius(self.T,self.H)

        #compute other variant-independent quantities here, (easy ones)
        
        if self.str_solver_variant == 'solvereal'
            dict_out = self.solve_real(dict_in)
        elif: self.str_solver_variant == 'solverealimag'
            dict_out = self.solve_real_imag(dict_in)
        elif: self.str_solver_variant == 'solvegroupsparse' or \
          self.str_solver_variant == 'solvegroupsparsefast'
            dict_out = self.solve_real_g(dict_in)
        elif: self.str_solver_variant == 'solvevbmm'
            dict_out = self.solve_vbmm(dict_in)
        elif: self.str_solver_variant == 'solvevbmmg'
            dict_out = self.solve_vbmm_g(dict_in)
        else:             
            raise Exception("no such MSIST solver variant " + self.str_solver_variant \
                            + " supported")
        return dict_out
    
    def solve_real(self,dict_in):
        Hhat = self.H.get_spectrum
        W = self.W
        Hhat_star = conj(Hhat)
        y_hat = dict_in['y_hat']
        x_n = dict_in['x_0']
        ary_alpha = dict_in['alpha']
        nu = dict_in['nu']
        epsilon = dict_in['epsilon']
        w_n = W * x_n
        #begin iterations here
        for n in np.arange(self.int_iterations):
            w_resid = W * ifftn(conj(Hhat_star) * (y_hat - Hhat * fftn(x_n)))
            for s in arange(1,ws_resid.int_subbands):
                w_n.set_subband(s, \
                  (ary_alpha[s]+(nu[n]^2) * 1/(w_n.get_subband(s)**2 + epsilon[n]^2))**(-1) * \
                  (ary_alpha[s]*w_n.get_subband(s)+w_resid.get_subband(s))
            x_n = ~W * w_n
            w_n = W * x_n #reprojection, to put our iterate in the range space, prevent drifting
        dict_in['x_n'] = x_n
        dict_in['w_n'] = w_n
        return dict_in

    def solve_vbmm(self,dict_in):
        Hhat = self.H.get_spectrum
        W = self.W
        Hhat_star = conj(Hhat)
        y_hat = dict_in['y_hat']
        x_n = dict_in['x_0']
        ary_alpha = dict_in['alpha']
        nu = dict_in['nu']
        w_n = W * x_n
        #begin iterations here
        for n in np.arange(self.int_iterations):
            w_resid = W * ifftn(conj(Hhat_star) * (y_hat - Hhat * fftn(x_n)))
            for s in arange(1,ws_resid.int_subbands):
                w_n.set_subband(s, \
                                
            x_n = ~W * w_n
            w_n = W * x_n #reprojection, to put our iterate in the range space, prevent drifting
        dict_in['x_n'] = x_n
        dict_in['w_n'] = w_n
        return dict_in
    
    
    class Factory:
        def create(self,ps_parameters,str_section):
            return MSIST(ps_parameters,str_section)

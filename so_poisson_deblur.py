#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj, sqrt, ones
from numpy import abs as nabs, exp, maximum as nmax
from py_utils.signal_utilities.ws import WS
import py_utils.signal_utilities.sig_utils as su
from py_solvers.solver import Solver
from py_operators.operator import Operator
from py_operators.operator_comp import OperatorComp
from scipy.optimize import fmin_ncg as ncg
from numpy.linalg import norm
from libtiff import TIFFimage

from py_utils.signal_utilities.sig_utils import crop

#profiling and debugging stuff
from py_utils.timer import Timer
import pdb

class PoissonDeblur(Solver):
    """
    Solver which performs the Poisson Deblurring Algorithm (for widefield microscopy deconvolution)
    """
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(PoissonDeblur,self).__init__(ps_parameters,str_section)
        self.str_solver_variant = self.get_val('solvervariant',False)
        self.str_sparse_pen = self.get_val('sparsepenalty',False)
        self.alpha = None
        self.H = OperatorComp(ps_parameters,self.get_val('modalities',False))
        self.H = self.H.ls_operators[0] #assume we just have one transform
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
        #ground truth data
        x_n = dict_in['x_0'].copy()
        del dict_in['x_0']
        b = dict_in['b']#background
        sigma_sq = dict_in['noisevariance']
        dict_in['x_n'] = x_n
        w_n = W * x_n
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

        S_n = WS(np.zeros(w_n.ary_lowpass.shape),(w_n.one_subband(0)).tup_coeffs) #initialize the variance matrix as a ws object
        dict_in['w_n'] = w_n
        if self.str_sparse_pen == 'l0rl2': #msist
            epsilon,nu = self.get_epsilon_nu()
            dict_in['nu_sq'] = nu**2
            dict_in['epsilon_sq'] = epsilon**2
        elif self.str_sparse_pen == 'vbmm': #vbmm    
            nu = self.get_val('nustart',True) * np.ones(self.int_iterations,)
            dict_in['nu_sq'] = nu**2
            p_a = self.get_val('p_a',True)
            p_b_0 = self.get_val('p_b_0',True)
            p_k = self.get_val('p_k',True)
            p_theta = self.get_val('p_theta',True)
            b_n = WS(np.zeros(w_n.ary_lowpass.shape),(w_n.one_subband(0)).tup_coeffs)
            for s in arange(w_n.int_subbands):
                b_n.set_subband(s, p_b_0)
        else:
            raise ValueError("This is not a valid sparse regularizer...: " + self.str_sparse_pen)
        #begin iterations here
        self.results.update(dict_in)
        yb = dict_in['y'] - b
        w_y = (W*dict_in['y_padded'])
        y_scaling = w_y.tup_scaling
        y_lowpass = w_y.ary_lowpass
        #y_scaling = (W*dict_in['y']).tup_scaling #for toy problem only
        if self.alpha.__class__.__name__ != 'ndarray':
            self.alpha = su.spectral_radius(self.W,self.H,(136,136,64))
        alpha = self.alpha
        alpha = 2.0*alpha
        #alpha = 0.012 * np.ones(w_n.int_subbands,)
        alpha[0] = 1
        print alpha
        for n in np.arange(self.int_iterations):
            #save current iterate
            if np.mod(n,10)==0:
                output_tiff = TIFFimage(dict_in['x_n'].astype('float32'))
                output_tiff.write_file('p0_itn' + str(n) + '.tif')
            f_resid = yb - H * x_n
            w_resid = W * (~H * f_resid)
            for s in arange(w_n.int_subbands):
                #variance estimate
                if s==0:
                    ary_variance = y_lowpass
                else:    
                    int_level, int_orientation = w_n.lev_ori_from_subband(s)
                    var_image = y_scaling[int_level]
                    if x_n.ndim==2:
                        ds_var_image = 1/4.0*(var_image[0::2,0::2]+var_image[1::2,0::2]+var_image[0::2,1::2]+var_image[1::2,1::2])
                        ary_variance = ((1.0/2.0)**(int_level))*(ds_var_image)
                    else:    
                        ds_var_image = 1/8.0*(var_image[0::2,0::2,0::2]+var_image[1::2,0::2,0::2]+var_image[0::2,1::2,0::2]+var_image[1::2,1::2,0::2] + \
                                              var_image[0::2,0::2,1::2]+var_image[1::2,0::2,1::2]+var_image[0::2,1::2,1::2]+var_image[1::2,1::2,1::2])
                        ary_variance = ((1.0/(2.0*np.sqrt(2)))**(int_level))*(ds_var_image)
                    
                    #print 'level: ' + str(int_level) + ' scaling mean: ' + str(np.mean(ary_variance))
                    ary_variance=0
                if self.str_sparse_pen == 'l0rl2': #msist
                    S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                elif self.str_sparse_pen == 'vbmm': #vbmm    
                    if n == 0:
                        sigma = 0
                    else:
                        sigma = (1.0 / (nu[n]**2 + ary_variance) * alpha[s] + S_n.get_subband(s))**(-1)
                    S_n.set_subband(s, (g_i + 2.0 * p_a) / (nabs(w_n.get_subband(s))**2 + sigma + 2.0 * b_n.get_subband(s)))
                    b_n.set_subband(s, (p_k + p_a) / (S_n.get_subband(s) + p_theta))
                else:
                    raise ValueError('no such sparse penalty')
                #update current solution
                #pdb.set_trace()
                w_n.set_subband(s, \
                  (alpha[s] * w_n.get_subband(s) + w_resid.get_subband(s)) / \
                  (alpha[s] + (ary_variance + nu[n]**2) * S_n.get_subband(s)))
            x_n = ~W * w_n
            print 'x_n min: ' + str(np.min(x_n))
            print 'x_n max: ' + str(np.max(x_n))

            x_n[x_n<b] = 0 #correct for negative values

            w_n = W * x_n #reprojection, to put our iterate in the range space, prevent drifting
            dict_in['x_n'] = x_n
            dict_in['w_n'] = w_n
            #update results
            self.results.update(dict_in)
            #save current iterate

        return dict_in

    class Factory:
        def create(self,ps_parameters,str_section):
            return PoissonDeblur(ps_parameters,str_section)

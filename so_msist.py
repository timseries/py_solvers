#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj, sqrt,median, abs as nabs, exp, maximum as nmax
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
        self.alpha_method = self.get_val('alphamethod',False)
        if self.alpha_method=='':
            self.alpha_method = 'spectrum'
        self.str_group_structure = self.get_val('grouptypes',False)
        
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
            self.alpha = su.spectral_radius(self.W,self.H,dict_in['x_0'].shape,self.alpha_method)
        alpha = self.alpha    
        w_n = W * x_n
        S_n = WS(np.zeros(w_n.ary_lowpass.shape),(w_n.one_subband(0)).tup_coeffs) #initialize the variance matrix as a ws object
        #every variant does these
        dict_in['w_n'] = w_n
        epsilon,nu = self.get_epsilon_nu()
        dict_in['nu_sq'] = nu**2
        dict_in['epsilon_sq'] = epsilon**2
        #l0rl2_bivar specific
        if self.str_sparse_pen == 'l0rl2_bivar':    
            #the thresholds which is saved 
            w_tilde = WS(np.zeros(w_n.ary_lowpass.shape),
                         (w_n.one_subband(0)).tup_coeffs)

        #vbmm specific
        if (self.str_sparse_pen == 'vbmm' or  #vbmm    
            self.str_sparse_pen == 'vbmm_hmt'):
            b_n = WS(np.zeros(w_n.ary_lowpass.shape),
                     (w_n.one_subband(0)).tup_coeffs)

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
        #begin iterations here
        self.results.update(dict_in)
        adj_factor = 1.3
        for n in np.arange(self.int_iterations):
            # if np.mod(n,100)==0:
            H.set_output_fourier(True)
            f_resid = ifftn(y_hat - H * x_n)
            H.set_output_fourier(False)
            w_resid = W * (~H * f_resid)

            if self.str_sparse_pen == 'l0rl2_bivar' and n==0:
                hh = (nabs(w_n.get_subband(2))).flatten()
                sigsq_n = su.mad(hh)/.6745 #eq 7, Sendur BSWLVE paper, why isn't this squared?
                sig_n = sqrt(sigsq_n)
                sqrt3=sqrt(3.0)
            for s in xrange(w_n.int_subbands-1,-1,-1):
                #variance estimate
                if self.str_sparse_pen == 'l0rl2': #msist
                    S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                    
                elif self.str_sparse_pen == 'l0rl2_bivar':
                    if s > 0:    
                        s_parent_us = nabs(self.get_upsampled_parent(s,w_n))**2
                        s_child = nabs(w_n.get_subband(s))**2
                        yi,yi_mask = su.get_neighborhoods(s_child,1) #eq 8, Sendur BSWLVE paper
                        s_child_norm = sqrt(s_parent_us + s_child)
                        if n==0:#np.mod(n,20)==0:    
                            sigsq_y = np.sum(yi,axis=yi.ndim-1)/np.sum(yi_mask,axis=yi.ndim-1)#still eq 8...
                            sig = sqrt(np.maximum(sqrt(sigsq_y)-sig_n,0))
                            w_tilde.set_subband(s, sqrt3*sig_n/sig) #the thresholding fn
                        thresh = np.maximum(s_child_norm - w_tilde.get_subband(s),0)/s_child_norm #eq 5
                        
                        if np.mod(n,1)==0:    
                            S_n.set_subband(s,(1.0 / 
                                               ((1.0 / g_i) *  
                                                nabs(thresh * w_n.get_subband(s))**2 + (epsilon[n]**2))))
                        else:
                            S_n.set_subband(s,(1.0 / 
                                               ((1.0 / g_i) * 
                                                nabs(w_n.get_subband(s))**2 + (epsilon[n]**2))))
                    else:
                        S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                        
                elif self.str_sparse_pen == 'vbmm': #vbmm    
                    if n == 0:
                        sigma_n = 0
                    else:
                        sigma_n = (1.0 / nu[n]**2 * alpha[s] + S_n.get_subband(s))**(-1)
                    S_n.set_subband(s, (g_i + 2.0 * p_a) / 
                                    (nabs(w_n.get_subband(s))**2 + 
                                     sigma_n + 2.0 * b_n.get_subband(s)))
                    b_n.set_subband(s, (p_k + p_a) / 
                                    (S_n.get_subband(s) + p_theta))
                elif self.str_sparse_pen == 'vbmm_hmt': #vbmm    
                    if n == 0:
                        sigma_n = 0
                    else:
                        sigma_n = (1.0 / nu[n]**2 * alpha[s] + S_n.get_subband(s))**(-1)
                    # if s < w_n.int_subbands - w_n.int_orientations and s > 0:    
                    if s > 0:    
                        s_parent_us = self.get_upsampled_parent(s,w_n)
                        #small_var_mask = s_parent_us**2 < 10*np.mean(s_parent_us**2)
                        # alpha_dec = small_var_mask * 3.1 + (1-small_var_mask) * 2.25
                        alpha_dec = 2.25
                        # alpha_dec = 3.1
                        s_child_en = np.abs(w_n.get_subband(s))**2
                        s_child_sz = s_child_en.shape
                        s_child_en = 1/4.0*(s_child_en[0::2,0::2] +
                                          s_child_en[1::2,0::2] +
                                          s_child_en[0::2,1::2] +
                                          s_child_en[1::2,1::2])
                        s_child_en_avg = np.zeros(s_child_sz)
                        s_child_en_avg[0::2,0::2] = s_child_en
                        s_child_en_avg[1::2,0::2] = s_child_en
                        s_child_en_avg[0::2,1::2] = s_child_en
                        s_child_en_avg[1::2,1::2] = s_child_en
                        if np.mod(n,100)==0:#n==0:
                            b_n.set_subband(s,ary_a[s] * 1/5.0*(4.0*s_child_en_avg+np.abs(s_parent_us)**2))
                            if s==10:
                                print 'b estimate'
                        # b_n.set_subband(s,ary_a[s] * 1/2.0*(nabs(w_n.get_subband(s))**2+np.abs(s_parent_us)**2))
                        # b_n.set_subband(s,ary_a[s] *  (2**(-alpha_dec)) * (np.abs(s_parent_us)**2))
                        S_n.set_subband(s, (g_i + 2.0 *  ary_a[s]) / 
                                        (nabs(w_n.get_subband(s))**2 + sigma_n + 
                                         2.0 * b_n.get_subband(s)))
                    else: #no parents, so generate fixed-param gammas
                        b_n.set_subband(s, (p_k + ary_a[s]) / 
                                        (S_n.get_subband(s) + p_theta))
                        S_n.set_subband(s, (g_i + 2.0 * ary_a[s]) / 
                                        (nabs(w_n.get_subband(s))**2 + sigma_n +
                                         2.0 * b_n.get_subband(s)))
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
            epsilon = epsilon_start*np.ones(self.int_iterations,)
            nu = nu_start*np.ones(self.int_iterations,)
        else:
            raise Exception('no such continuation parameter rule')
        return epsilon,nu

    def get_gamma_shapes2(self, ws_coeffs):
        """Use maximum likelihood to obtain the shape parameters 
        for each a gamma distribution.

        Return:
          ary_a (ndarray): a 1-d array of shape parameters
        """
        ary_a = np.zeros(ws_coeffs.int_subbands,)
        for s in xrange(1,ws_coeffs.int_subbands):
            subband = ws_coeffs.get_subband(s).flatten()
            subband_ri = np.concatenate((subband.real,subband.imag))
            fit_nu, fit_loc, fit_scale = ss.t.fit(subband_ri,floc=0)
            print fit_scale
            ary_a[s] = fit_nu / 2
        print ary_a    
        return ary_a    

    def get_gamma_shapes(self, ws_coeffs):
        """Use maximum likelihood to obtain the shape parameters 
        for each a gamma distribution.

        Return:
          ary_a (ndarray): a 1-d array of shape parameters
        """
        ary_a = np.zeros(ws_coeffs.int_subbands,)
        # for s in xrange(1,ws_coeffs.int_subbands,6):
        for s in xrange(1,ws_coeffs.int_subbands,6):
            for s2 in xrange(s,s+6):
                subband = ws_coeffs.get_subband(s2).flatten()
                subband_ri = np.concatenate((subband.real,subband.imag))
                if s2==s:
                    subband_tot=np.zeros(subband_ri.size*4)
                    subband_tot_45=np.zeros(subband_ri.size*2)
                    index=0
                    index_45=0
                if np.mod(s2-2,6)==0 or np.mod(s2-5,6)==0:
                    subband_tot_45[index_45:index_45+subband_ri.size]=subband_ri.copy()
                    index_45=index_45+subband_ri.size
                else:    
                    subband_tot[index:index+subband_ri.size]=subband_ri.copy()
                    index=index+subband_ri.size
            fit_nu, fit_loc, fit_scale = ss.t.fit(subband_tot)
            # print fit_scale
            ary_a[s] = fit_nu / 2
            ary_a[s+2:s+4] = fit_nu / 2
            ary_a[s+5] = fit_nu / 2
            fit_nu, fit_loc, fit_scale = ss.t.fit(subband_tot_45)
            # print fit_scale
            ary_a[s+1] = fit_nu / 2
            ary_a[s+4] = fit_nu / 2
        print ary_a    
        return ary_a    

    def get_upsampled_parent(self,s,w_n):
        """Return the upsampled parent layer of subband s
        """
        if s+w_n.int_orientations>=w_n.int_subbands:
            #we actually need to downsample in this case
            subband_index = 0
        else:
            subband_index = s+w_n.int_orientations
        s_parent = w_n.get_subband(subband_index)
        if subband_index==0:
            s_parent_us=1/4.0*(s_parent[0::2,0::2] + 
                               s_parent[0::2,1::2] + 
                               s_parent[1::2,0::2] + 
                               s_parent[1::2,1::2])
        else:
            s_parent_us=np.zeros((2*s_parent.shape[0],2*s_parent.shape[1]))
            #todo: generalize this to arbitrary dimensions
            s_parent_us[0::2,0::2]=s_parent
            s_parent_us[0::2,1::2]=s_parent
            s_parent_us[1::2,0::2]=s_parent
            s_parent_us[1::2,1::2]=s_parent
        return s_parent_us

    class Factory:
        def create(self,ps_params,str_section):
            return MSIST(ps_params,str_section)

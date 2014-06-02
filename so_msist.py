#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj, sqrt,median, abs as nabs, exp, maximum as nmax
from numpy.fft import fftn, ifftn
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

from py_utils.signal_utilities.ws import WS
import py_utils.signal_utilities.sig_utils as su
from py_solvers.solver import Solver
from py_operators.operator import Operator
from py_operators.operator_comp import OperatorComp
from py_utils.section_factory import SectionFactory as sf

import scipy.stats as ss

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
        self.spatial_threshold = self.get_val('spatialthreshold',True)#for poisson deblurring only
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
        x_n = dict_in['x_0'].copy() #initial solution
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
        #initialize the precision matrix
        S_n = WS(np.zeros(w_n.ary_lowpass.shape),(w_n.one_subband(0)).tup_coeffs) #initialize the variance matrix as a ws object
        #every variant does these next steps
        dict_in['w_n'] = w_n
        epsilon,nu = self.get_epsilon_nu()
        dict_in['nu_sq'] = nu**2
        dict_in['epsilon_sq'] = epsilon**2
        #l0rl2_bivar specific
        if self.str_sparse_pen == 'l0rl2_bivar':    
            #the thresholds which is saved 
            w_tilde = WS(np.zeros(w_n.ary_lowpass.shape),
                         (w_n.one_subband(0)).tup_coeffs)
            sqrt3=sqrt(3.0)
        #l0rl2_group specific
        if self.str_sparse_pen == 'l0rl2_group':
            #initialize the cluster averaging matrix, A
            tau = self.get_val('tau',True)
            #the non-overlapping space is just a list of ws objects
            #copy the initial ws object here
            A = sf.create_section(self.ps_parameters,self.get_val('clusteraverage',False))
            G = sf.create_section(self.ps_parameters,self.get_val('groupaverage',False))
            ls_w_n_hat = [WS(w_n.ary_lowpass,w_n.tup_coeffs)
                          for int_dup in xrange(A.duplicates)]
            dict_in['ls_w_n_hat'] = ls_w_n_hat
            ls_S_hat_n = [WS(S_n.ary_lowpass,S_n.tup_coeffs)
                          for int_dup in xrange(A.duplicates)]
            dict_in['ls_S_hat_n'] = ls_S_hat_n
            w_n_bar = WS(w_n.ary_lowpass,w_n.tup_coeffs)
            dict_in['w_n_bar'] = w_n_bar
            A*ls_w_n_hat #to initialize A and G
            G*ls_w_n_hat
            #precompute tau^2*AtA
            tausq_AtA = (tau**2)*(A.A.transpose()*A.A)
            del S_n #don't need this, since we are using S_hat_n
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
            adj_factor = 1.3
    
        #poisson-corrupted gaussiang noise, using the scaling coefficients in the regularization (MSIST-P)
        if (self.str_sparse_pen == 'poisson_deblur'):
            #need a 0-padded y to get the right size for the scaling coefficients
            b = dict_in['b']
            y_hat = fftn(dict_in['y'] - b) #need a different yhat for the iterations...
            w_y = (W*dict_in['y_padded'])
            dict_in['x_n']=su.crop_center(x_n,dict_in['y'].shape)
            w_y_scaling_coeffs=w_y.downsample_scaling()
        if (self.str_sparse_pen == 'poisson_deblur_sp'):
            #need a 0-padded y to get the right size for the scaling coefficients
            b = dict_in['b']
            y_hat = fftn(dict_in['y'] - b) #need a different yhat for the iterations...
            w_y = (W*dict_in['y_padded'])
            dict_in['x_n']=su.crop_center(x_n,dict_in['y'].shape)
            w_y_scaling_coeffs=w_y.downsample_scaling()
        #perform initial (n=0) update of the results
        self.results.update(dict_in)
        print 'Finished itn: n=' + str(0)
        #begin iterations here for the MSIST algorithm
        for n in np.arange(self.int_iterations):
            #doing fourier domain operation explicitly here, this save some transforms in the next step
            H.set_output_fourier(True)
            #Landweber difference
            f_resid = ifftn(y_hat - H * x_n)
            if (self.str_sparse_pen == 'poisson_deblur_sp'):#spatial domain thresholding goes here
                f_resid/=(dict_in['y']+nu[n]**2)
            H.set_output_fourier(False)
            #Landweber projection back to spatial domain
            w_resid = W * (~H * f_resid)
            #the bivariate shrinkage penalty in l0rl2 (coefficient neighborhood averags)
            if self.str_sparse_pen == 'l0rl2_bivar' and n==0:
                sigsq_n = self.get_val('nustop', True)**2
                sig_n = sqrt(sigsq_n)
            #update each of the wavelet subbands separatly, starting at the 
            #coarsest and working to finer scales
            #While this can be done as a single vector operation, that would
            #require a sparse version of \Lambda_{\alpha} with many entries copied
            #without much performance gain over this for loop
            ary_p_var=0 #this is zero for all variance, except the poisson deblurring
            for s in xrange(w_n.int_subbands-1,-1,-1):
                #variance estimate depends on the version of msist {l0rl2,l0rl2_bivar,vbmm,vbmm_hmt,poisson_deblur)
                #this gives different sparse penalties
                if self.str_sparse_pen == 'l0rl2': #basic msist algorithm
                    S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                elif self.str_sparse_pen[:7] == 'poisson':
                    if 1: #self.str_sparse_pen == 'poisson_deblur': #msist with poisson-corrupted noise, 
                        if s==0:
                            ary_p_var = w_y.ary_lowpass
                        else:
                            int_level, int_orientation = w_n.lev_ori_from_subband(s)
                            ary_p_var = w_y_scaling_coeffs[int_level]
                        ary_p_var[ary_p_var<=0]=.01
                    S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                elif self.str_sparse_pen == 'l0rl2_bivar':
                    if s > 0:    
                        s_parent_us = nabs(w_n.get_upsampled_parent(s))**2
                        s_child = nabs(w_n.get_subband(s))**2
                        yi,yi_mask = su.get_neighborhoods(s_child,1) #eq 8, Sendur BSWLVE paper, yzhang code doesn't do this...
                        s_child_norm = sqrt(s_parent_us + s_child)
                        sigsq_y = np.sum(yi,axis=yi.ndim-1)/np.sum(yi_mask,axis=yi.ndim-1)#still eq 8...
                        sig = sqrt(np.maximum(sigsq_y-sigsq_n,0))
                        w_tilde.set_subband(s, sqrt3*sigsq_n/sig) #the thresholding fn
                        thresh = np.maximum(s_child_norm - w_tilde.get_subband(s),0)/s_child_norm #eq 5
                        if np.mod(n,2)==0:    #update with the bivariate thresholded coefficients on every other iteration
                            S_n.set_subband(s,(1.0 / 
                                               ((1.0 / g_i) *  
                                                nabs(thresh * w_n.get_subband(s))**2 + (epsilon[n]**2))))
                        else:
                            S_n.set_subband(s,(1.0 / 
                                               ((1.0 / g_i) * 
                                                nabs(w_n.get_subband(s))**2 + (epsilon[n]**2))))
                    else:
                        S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                elif self.str_sparse_pen == 'l0rl2_group': #msist-g
                    #dummy update...do updates later
                    ls_S_hat_n=ls_S_hat_n
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
                        w_parent_us = w_n.get_upsampled_parent(s)
                        alpha_dec = 2.25
                        if s > S_n.int_orientations:
                            s_child = S_n.subband_group_sum(s-S_n.int_orientations,'children')
                            b_child = b_n.subband_group_sum(s-S_n.int_orientations,'children')
                        else:
                            s_child = 0 
                            b_child = 0   
                        if s < S_n.int_subbands - S_n.int_orientations:
                            ap = ary_a[s+S_n.int_orientations]
                        else:
                            ap = .5
                        w_en_avg = w_n.subband_group_sum(s,'parent_children')
                        # S_n.set_subband(s, (g_i + 2.0 *  ary_a[s]) / 
                        #                 (nabs(w_n.get_subband(s))**2 + sigma_n + 
                        #                  2.0 * b_n.get_subband(s)))
                        #the standard vbmm b with estimated shape parameter,works with the parent+4 children b
                        S_n.set_subband(s, (g_i + 2.0 *  ary_a[s]) / 
                                        (nabs(w_n.get_subband(s))**2 + sigma_n + 
                                         2.0 * b_n.get_subband(s)))
                        #the vbmm hmt model for s,b, ngk
                        # S_n.set_subband(s, (g_i + 2.0 * ap + ary_a[s]) / 
                        #                 (nabs(w_n.get_subband(s))**2 + sigma_n + 
                        #                  2.0 * ((2**(-alpha_dec))*b_child + b_n.get_subband(s))))
                        # the parent+4 children bs
                        b_n.set_subband(s,ary_a[s] * w_en_avg)
                        #the vbmm hmt model for s,b, ngk
                        # b_n.set_subband(s,(4*ary_a[s]+ap) /
                        #                 (s_child + 2**(-alpha_dec) * S_n.get_subband(s)))
                        #other b estimators
                        # b_n.set_subband(s,ary_a[s] * 1/2.0*(nabs(w_n.get_subband(s))**2+np.abs(w_parent_us)**2))
                        # b_n.set_subband(s,ary_a[s] *  (2**(-alpha_dec)) * (np.abs(w_parent_us)**2))
                    else: #no parents, so generate fixed-param gammas
                        S_n.set_subband(s, (g_i + 2.0 * ary_a[s]) / 
                                        (nabs(w_n.get_subband(s))**2 + sigma_n +
                                         2.0 * b_n.get_subband(s)))
                        b_n.set_subband(s, (p_k + ary_a[s]) / 
                                        (S_n.get_subband(s) + p_theta))
                else:
                    raise ValueError('no such solver variant')
                #update current solution
                if (self.str_sparse_pen == 'poisson_deblur_sp'):#spatial domain thresholding goes here
                    w_n.set_subband(s, \
                      (alpha[s] * w_n.get_subband(s) + w_resid.get_subband(s)) / \
                      (alpha[s] + S_n.get_subband(s)))
                elif (self.str_sparse_pen == 'poisson_deblur'):
                    w_n.set_subband(s, \
                      (alpha[s] * w_n.get_subband(s) + w_resid.get_subband(s)) / \
                      (alpha[s] + (nu[n]**2+ary_p_var) * S_n.get_subband(s)))
                elif (self.str_sparse_pen == 'l0rl2_group'):
                    ls_S_hat_n=G*[w_n_hat.energy() for w_n_hat in ls_w_n_hat] #eq 11
                    ls_S_hat_n=[S_hat_n.sum(epsilon**2) for S_hat_n in ls_S_hat_n] #eq 11
                    ls_S_hat_n=[S_hat_n.invert() for S_hat_n in ls_S_hat_n] #eq 19
                    ary_S_hat_n=su.flatten_list(ls_S_hat_n)
                    ary_diag=np.arange(S_hat_n_length)
                    S_hat_n_sz=ary_S_hat_n.size
                    S_hat_n_csr=csr_matrix((su.flatten_list(),(ary_diag,ary_diag)), shape=(S_hat_n_sz,S_hat_n_sz))
                    ls_w_n_hat=(~A)*((tau**2)*w_n) #eq 21
                    w_n_hat_flat=su.flatten_list(ls_w_n_hat)
                    w_n_hat_flat=inv(tausq_AtA + (nu**2)*S_hat_n_csr)*w_n_hat_flat #eq 21 contd
                    ls_w_n_hat_unflat=su.unflatten_list(w_n_hat_flat)
                    ws_dummy=WS(np.zeros(w_n.ary_lowpass.shape),(w_n.one_subband(0)).tup_coeffs)
                    ws_dummy.flatten()
                    #the +0 in the next line returns a new WS object...
                    ls_w_n_hat=[ws_dummy.unflatten(w_n_hat_unflat)+0 for w_n_hat_unflat in ls_w_n_hat_unflat]
                    w_n_bar=A*ls_w_n_hat #eq 13
                    w_n.set_subband(s, 
                      (alpha[s] * w_n.get_subband(s) + w_resid.get_subband(s) + 
                       (tau**2)*w_n_bar.get_subband(s)) /
                       (alpha[s] + tau**2))
                    #reprojection for the duplicated vectors
                    ls_w_n_hat = [W*((~W)*w_n_hat) for w_n_hat in ls_w_n_hat]
                else:
                    w_n.set_subband(s, \
                      (alpha[s] * w_n.get_subband(s) + w_resid.get_subband(s)) / \
                      (alpha[s] + (nu[n]**2) * S_n.get_subband(s)))
                #end updating subbands   
            x_n = ~W * w_n
            if self.str_sparse_pen=='poisson_deblur':
                #need to remove the invalid border of this iterate
                x_n=su.crop_center(x_n,dict_in['y'].shape)
                #remove the background
                if self.spatial_threshold:
                    x_n[x_n<b]=0.0
            dict_in['x_n'] = x_n
            #reprojection, to put our iterate in the transform range space, prevent 'drifting'
            #need to reset the border of this iterate for the next implicit convolution
            if self.str_sparse_pen=='poisson_deblur':
                x_n=su.pad_center(x_n,dict_in['x_0'].shape)
            w_n = W * x_n
            dict_in['w_n'] = w_n
            #update results
            self.results.update(dict_in)
            print 'Finished itn: n=' + str(n+1)
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


    class Factory:
        def create(self,ps_params,str_section):
            return MSIST(ps_params,str_section)

#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj, sqrt,median, abs as nabs, exp, maximum as nmax, angle
from numpy.fft import fftn, ifftn
from numpy.linalg import norm
from scipy.sparse import csr_matrix

from py_utils.signal_utilities.ws import WS
import py_utils.signal_utilities.sig_utils as su
from py_solvers.solver import Solver
from py_operators.operator import Operator
from py_operators.operator_comp import OperatorComp
from py_utils.section_factory import SectionFactory as sf

import scipy.stats as ss

import os
import cPickle

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
        self.H = OperatorComp(ps_params,self.get_val('modalities',False))
        if len(self.H.ls_ops)==1: #avoid slow 'eval' in OperatorComp
            self.H = self.H.ls_ops[0] 
        self.W = OperatorComp(ps_params,self.get_val('transforms',False))
        if len(self.W.ls_ops)==1: #avoid slow 'eval' in OperatorComp
            self.W = self.W.ls_ops[0] 
        self.alpha = self.get_val('alpha',True)
        self.alpha_method = self.get_val('alphamethod',False)
        self.spatial_threshold = self.get_val('spatialthreshold',True)#for poisson deblurring only
        if self.alpha_method=='':
            self.alpha_method = 'spectrum'
        self.str_group_structure = self.get_val('grouptypes',False)
        self.phase_encoded = self.get_val('phaseencoded',True)
        
    def solve(self,dict_in):
        """
        Takes an input object (ground truth, forward model observation, metrics required)
        Returns a solution object based on the solver this object was instantiated with.
        """
        super(MSIST,self).solve()
        H = self.H
        W = self.W
        #input data 
        if not H.output_fourier:
            y_hat = fftn(dict_in['y'])
        else:
            y_hat = dict_in['y'] #y is already in the Fourier domain (NMR/MRI)
        x_n = dict_in['x_0'].copy() #initial solution
        dict_in['x_n'] = x_n
        if self.str_sparse_pen[-4:] == 'cplx':    
            dict_in['theta_n'] = angle(x_n)
            dict_in['magnitude_n'] = nabs(x_n)
        #input parameters and initialization
        if self.alpha.__class__.__name__ != 'ndarray':
            self.alpha = su.spectral_radius(self.W,self.H,dict_in['x_0'].shape,self.alpha_method)
        alpha = self.alpha    
        if self.str_sparse_pen[-4:] == 'cplx':    
            w_n = [W * x_n.real, W * x_n.imag]
            g_i = 2 * (w_n[0].is_wavelet_complex() + 1)
        else:
            w_n = [W * x_n]
            g_i = (w_n[0].is_wavelet_complex() + 1)
        g_i_half = g_i/2 #used to normalize computation of S
        w_n_it = xrange(g_i_half) #iterator for w_n list (real vs real/complex w_n)
        #initialize the precision matrix
        S_n = w_n[0]*0 #initialize the variance matrix as a ws object
        dict_in['w_n'] = w_n
        epsilon,nu = self.get_epsilon_nu()
        dict_in['nu_sq'] = nu**2
        dict_in['epsilon_sq'] = epsilon**2
        if self.str_sparse_pen == 'l0rl2_bivar':    
            w_tilde = w_n[0]*0
            sqrt3=sqrt(3.0)
        if self.str_sparse_pen[:11] == 'l0rl2_group':
            tau = self.get_val('tau',True)
            #the non-overlapping space is just a list of ws objects
            #copy the initial ws object here
            A = sf.create_section(self.ps_parameters,self.get_val('clusteraverage',False))
            G = sf.create_section(self.ps_parameters,self.get_val('groupaverage',False))
            #initialize A and G by doing dummy multiplies (initialized with parameters of the master vector)
            A.init_csr_avg(w_n[0])
            G.init_csr_avg(w_n[0])
            dup_it = xrange(A.duplicates) # iterator for duplicate variable space
            ls_S_hat_n = [((sum([w_n[ix_].energy() for ix_ in w_n_it]) / g_i_half) + epsilon[0]**2).invert()
                          for int_dup in dup_it]
            ls_w_hat_n = [[w_n[ix_]*1 for j in dup_it] for ix_ in w_n_it]
            w_bar_n = [w_n[ix_]*1 for ix_ in w_n_it]
            #now using the structure of A, initialize w_n_hat by
            #copying the elements of w_n in the correct places
            #precompute tau^2*AtA
            A_row_indices=np.nonzero(A.csr_avg)[0]
            A_col_indices=np.nonzero(A.csr_avg)[1]
            D=csr_matrix((np.ones(A_col_indices.size,),(A_row_indices,A_col_indices)),shape=A.csr_avg.shape)
            ls_S_hat_sup=su.unflatten_list(D.transpose()*((w_n[0]*0+1).flatten()),A.duplicates)
            ls_S_hat_sup=[(w_n[0]*0).unflatten(S_hat_n_sup) for S_hat_n_sup in ls_S_hat_sup]
            ls_S_hat_sup=[S_hat_n_sup.nonzero() for S_hat_n_sup in ls_S_hat_sup]
            tausq_AtA = (tau**2)*(A.csr_avg.transpose()*A.csr_avg).tocsr()
            #initialize S_hat_bar parameters for efficient matrix inverses
            Shatbar_p_filename=A.file_path.split('.pkl')[0]+'Shatbar.pkl'
            if not os.path.isfile(Shatbar_p_filename):
                dict_in['col_offset']=A.int_size
                S_hat_n_csr=su.flatten_list_to_csr(ls_S_hat_sup)
                su.inv_block_diag(tausq_AtA+S_hat_n_csr,dict_in)
                #save the file
                filehandler=open(Shatbar_p_filename, 'wb')
                cPickle.dump(dict_in['dict_bdiag'],filehandler)
                del S_hat_n_csr
            else:    
                filehandler=open(Shatbar_p_filename, 'rb')
                dict_in['dict_bdiag']=cPickle.load(filehandler)
            filehandler.close()    
            #store all of the l0rl2_group specific variables in the solver dict_in
            dict_in['ls_S_hat_n']=ls_S_hat_n
            dict_in['ls_w_hat_n']=ls_w_hat_n
            dict_in['w_bar_n']=w_bar_n
            dict_in['G']=G
            dict_in['A']=A
            dict_in['W']=W
            dict_in['tausq_AtA']=tausq_AtA
            dict_in['ls_S_hat_sup']=ls_S_hat_sup
            dict_in['w_n_it']=w_n_it
            dict_in['dup_it']=dup_it
            dict_in['ws_dummy']=w_n[0]*0
            dict_in['g_i_half']=g_i_half
            
            self.update_duplicates(dict_in,nu[0],epsilon[0],tau)    
            
            w_bar_n=dict_in['w_bar_n'] 
            ls_w_hat_n=dict_in['ls_w_hat_n']
            ls_S_hat_n=dict_in['ls_S_hat_n']
            del D #don't need this anymore, update rules only depend on A
        #vbmm specific
        if (self.str_sparse_pen == 'vbmm' or  #vbmm    
            self.str_sparse_pen == 'vbmm_hmt'):
            p_a = self.get_val('p_a',True)
            p_b_0 = self.get_val('p_b_0',True)
            p_k = self.get_val('p_k',True)
            p_theta = self.get_val('p_theta',True)
            p_c = self.get_val('p_c',True)
            p_d = self.get_val('p_d',True)
            b_n = w_n[0]*0
            if self.str_sparse_pen == 'vbmm_hmt':
                ary_a = self.get_gamma_shapes(W * dict_in['x_0'])
            for s in arange(w_n[0].int_subbands):
                b_n.set_subband(s, p_b_0)
            adj_factor = 1.3
    
        #poisson-corrupted gaussiang noise, using the scaling coefficients in the regularization (MSIST-P)
        if (self.str_sparse_pen[0:7] == 'poisson'):
            #need a 0-padded y to get the right size for the scaling coefficients
            b = dict_in['b']
            if not H.outputfourier:
                y_hat = fftn(dict_in['y'] - b) #need a different yhat for the iterations...
            else:
                y_hat = fftn(ifftn(dict_in['y']) - b)
            w_y = (W*dict_in['y_padded'])
            dict_in['x_n']=su.crop_center(x_n,dict_in['y'].shape)
            w_y_scaling_coeffs=w_y.downsample_scaling()
        #perform initial (n=0) update of the results
        self.results.update(dict_in)
        print 'Finished itn: n=' + str(0)
        #begin iterations here for the MSIST(-X) algorithm
        for n in np.arange(self.int_iterations):
            #doing fourier domain operation explicitly here, this save some transforms in the next step
            H.set_output_fourier(True) #force Fourier output to reduce number of ffts
            #Landweber difference
            if self.str_sparse_pen[-4:] == 'cplx': 
                f_resid = y_hat - H * x_n
            else:    
                f_resid = ifftn(y_hat - H * x_n)
                H.set_output_fourier(False)
            if (self.str_sparse_pen == 'poisson_deblur_sp'):#spatial domain thresholding goes here
                f_resid/=(dict_in['y']+nu[n]**2)
            #Landweber projection back to observation domain
            if self.str_sparse_pen[-4:] == 'cplx': 
                HtHf = (~H) * f_resid
                w_resid = [W * (HtHf).real, W * (HtHf).imag]
            else:
                w_resid = [W * ((~H) * f_resid)]
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
            if self.str_sparse_pen == 'l0rl2_group':
            #calculate S_hat_n, w_hat_n, and wb_bar (eqs 11, 19, and 13)
                self.update_duplicates(dict_in, nu[n], epsilon[n], tau)    
                w_bar_n=dict_in['w_bar_n'] 
                ls_w_hat_n=dict_in['ls_w_hat_n']
            #####################################################    
            #subband-adaptive subband update of precision matrix#
            #####################################################    
            if self.str_sparse_pen == 'l0rl2' or self.str_sparse_pen == 'l0rl2_cplx': #basic msist algorithm
                S_n = (sum([w_n[ix_].energy() for ix_ in w_n_it]) / g_i_half + epsilon[n]**2).invert()
            else:    
                for s in xrange(w_n[0].int_subbands-1,-1,-1):
                    #variance estimate depends on the version of msist 
                    #{l0rl2,l0rl2_bivar,l0rl2_group,vbmm,vbmm_hmt,poisson_deblur}
                    #this gives different sparse penalties
                    if self.str_sparse_pen[:7] == 'poisson':
                        if 1: #self.str_sparse_pen == 'poisson_deblur': #msist with poisson-corrupted noise, 
                            if s==0:
                                ary_p_var = w_y.ary_lowpass
                            else:
                                int_level, int_orientation = w_n[0].lev_ori_from_subband(s)
                                ary_p_var = w_y_scaling_coeffs[int_level]
                            ary_p_var[ary_p_var<=0]=.01
                        S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n[0].get_subband(s))**2 + epsilon[n]**2)))
                    elif self.str_sparse_pen == 'l0rl2_bivar':
                        if s > 0:    
                            s_parent_us = nabs(w_n[0].get_upsampled_parent(s))**2
                            s_child = nabs(w_n[0].get_subband(s))**2
                            yi,yi_mask = su.get_neighborhoods(s_child,1) #eq 8, Sendur BSWLVE paper, yzhang code doesn't do this...
                            s_child_norm = sqrt(s_parent_us + s_child)
                            sigsq_y = np.sum(yi,axis=yi.ndim-1)/np.sum(yi_mask,axis=yi.ndim-1)#still eq 8...
                            sig = sqrt(np.maximum(sigsq_y-sigsq_n,0))
                            w_tilde.set_subband(s, sqrt3*sigsq_n/sig) #the thresholding fn
                            thresh = np.maximum(s_child_norm - w_tilde.get_subband(s),0)/s_child_norm #eq 5
                            if np.mod(n,2)==0:    #update with the bivariate thresholded coefficients on every other iteration
                                S_n.set_subband(s,(1.0 / 
                                                   ((1.0 / g_i) *  
                                                    nabs(thresh * w_n[0].get_subband(s))**2 + (epsilon[n]**2))))
                            else:
                                S_n.set_subband(s,(1.0 / 
                                                   ((1.0 / g_i) * 
                                                    nabs(w_n[0].get_subband(s))**2 + (epsilon[n]**2))))
                        else:
                            S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n[0].get_subband(s))**2 + epsilon[n]**2)))
                    elif self.str_sparse_pen == 'l0rl2_group': #msist-g, Shat is updated before this loop
                        if s==0:
                            S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n[0].get_subband(s))**2 + epsilon[n]**2)))
                    elif self.str_sparse_pen == 'vbmm': #vbmm    
                        if n == 0:
                            sigma_n = 0
                        else:
                            sigma_n = (1.0 / nu[n]**2 * alpha[s] + S_n.get_subband(s))**(-1)
                        S_n.set_subband(s, (g_i + 2.0 * p_a) / 
                                        (nabs(w_n[0].get_subband(s))**2 + sigma_n + 2.0 * b_n.get_subband(s)))
                        b_n.set_subband(s, (p_k + p_a) / 
                                        (S_n.get_subband(s) + p_theta))
                    elif self.str_sparse_pen == 'vbmm_hmt': #vbmm    
                        if n == 0:
                            sigma_n = 0
                        else:
                            sigma_n = (1.0 / nu[n]**2 * alpha[s] + S_n.get_subband(s))**(-1)
                        # if s < w_n.int_subbands - w_n.int_orientations and s > 0:    
                        if s > 0:    
                            w_parent_us = w_n[0].get_upsampled_parent(s)
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
                            w_en_avg = w_n[0].subband_group_sum(s,'parent_children')
                            # S_n.set_subband(s, (g_i + 2.0 *  ary_a[s]) / 
                            #                 (nabs(w_n.get_subband(s))**2 + sigma_n + 
                            #                  2.0 * b_n.get_subband(s)))
                            #the standard vbmm b with estimated shape parameter,works with the parent+4 children b
                            S_n.set_subband(s, (g_i + 2.0 *  ary_a[s]) / 
                                            (nabs(w_n[0].get_subband(s))**2 + sigma_n + 
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
                                            (nabs(w_n[0].get_subband(s))**2 + sigma_n +
                                             2.0 * b_n.get_subband(s)))
                            b_n.set_subband(s, (p_k + ary_a[s]) / 
                                            (S_n.get_subband(s) + p_theta))
                    else:
                        raise ValueError('no such solver variant')
            #########################    
            #update current solution#
            #########################
            for s in xrange(w_n[0].int_subbands-1,-1,-1):    
                if (self.str_sparse_pen == 'poisson_deblur_sp'):#spatial domain thresholding goes here
                    w_n[0].set_subband(s, \
                      (alpha[s] * w_n[0].get_subband(s) + w_resid[0].get_subband(s)) / \
                      (alpha[s] + S_n.get_subband(s)))
                elif (self.str_sparse_pen == 'poisson_deblur'):
                    w_n[0].set_subband(s, \
                      (alpha[s] * w_n[0].get_subband(s) + w_resid[0].get_subband(s)) / \
                      (alpha[s] + (nu[n]**2+ary_p_var) * S_n.get_subband(s)))
                elif (self.str_sparse_pen[:11] == 'l0rl2_group'):
                    if s>0:
                        for ix_ in w_n_it:
                            w_n[ix_].set_subband(s,
                                            (alpha[s] * w_n[ix_].get_subband(s) + w_resid[ix_].get_subband(s) + 
                                             (tau**2)*w_bar_n[ix_].get_subband(s)) / (alpha[s] + tau**2))
                    else: #a standard msist update for the lowpass coeffs
                        for ix_ in w_n_it:
                            w_n[ix_].set_subband(s, \
                                            (alpha[s] * w_n[ix_].get_subband(s) + w_resid[ix_].get_subband(s)) / 
                                            (alpha[s] + (nu[n]**2) * S_n.get_subband(s)))
                else: 
                    for ix_ in w_n_it:
                        w_n[ix_].set_subband(s, 
                                           (alpha[s] * w_n[ix_].get_subband(s) + w_resid[ix_].get_subband(s)) / 
                                           (alpha[s] + (nu[n]**2) * S_n.get_subband(s)))
                #end updating subbands   
            #project back to spatial domain for any spatial domain operations    
            if self.str_sparse_pen[-4:] == 'cplx':    
                x_n = np.asfarray(~W * w_n[0],'complex128') 
                x_n += 1j*np.asfarray(~W * w_n[1],'complex128')
                m_n = nabs(x_n)
                theta_n = angle(x_n)
                if self.phase_encoded: #need to apply boundary conditions for phase encoded velocity 
                    theta_n = su.phase_unwrap(angle(x_n),dict_in['dict_global_lims'],dict_in['ls_vcorrect_secs'])
                    if self.get_val('iterationmask',True): #apply boundary conditions for phase encoded velocity 
                        theta_n *= dict_in['mask']
                        # m_n *= dict_in['mask']
                    x_n = m_n*exp(1j*theta_n)
            else:    
                x_n = ~W * w_n[0]
            #need to unpad x_n now for computing metrics
            if self.str_sparse_pen=='poisson_deblur':
                #need to remove the invalid border of this iterate
                x_n=su.crop_center(x_n,dict_in['y'].shape)
                #remove the background
                if self.spatial_threshold:
                    x_n[x_n<b]=0.0
            dict_in['x_n'] = x_n
            if self.str_sparse_pen[-4:] == 'cplx':    
                dict_in['theta_n'] = theta_n
                dict_in['magnitude_n'] = m_n
            #need to reset the border of this iterate for the next implicit convolution(msistp)
            if self.str_sparse_pen=='poisson_deblur':
                x_n=su.pad_center(x_n,dict_in['x_0'].shape)
            #reprojection (for all algorithms)
            if self.str_sparse_pen[-4:] == 'cplx':    
                w_n = [W * x_n.real, W * x_n.imag]
            else:
                w_n = [W * x_n]
            #reprojection for the duplicated variables of l0rl2_group variant
            if self.str_sparse_pen[:11] == 'l0rl2_group':
                ls_w_hat_n = [[ls_w_hat_n[ix_][j]*ls_S_hat_sup[j]+w_n[ix_]*((ls_S_hat_sup[j]+(-1))*(-1))
                              for j in dup_it] for ix_ in w_n_it]
                ls_w_hat_n = [[W*((~W)*w_hat_n) for w_hat_n in ls_w_hat_n[ix_]] for ix_ in w_n_it]
                w_bar_n = [W*((~W)*w_bar_n[ix_]) for ix_ in w_n_it]
                dict_in['w_bar_n'] = w_bar_n
                dict_in['ls_w_hat_n'] = ls_w_hat_n
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

    def update_duplicates(self, dict_in, nu, epsilon, tau):
        '''update the duplicate estimates in (the msist-g algorithm updates)
        '''
        #these dict_in members change
        ls_S_hat_n=dict_in['ls_S_hat_n']
        ls_w_hat_n=dict_in['ls_w_hat_n']
        w_bar_n=dict_in['w_bar_n']
        #the rest don't change
        w_n=dict_in['w_n']
        G=dict_in['G']
        A=dict_in['A']
        W=dict_in['W']
        ws_dummy=dict_in['ws_dummy']
        tausq_AtA=dict_in['tausq_AtA']
        ls_S_hat_sup=dict_in['ls_S_hat_sup']
        g_i_half = dict_in['g_i_half']
        w_n_it = dict_in['w_n_it']
        dup_it = dict_in['dup_it']
        #perform the updats
        ls_S_hat_n=G*[sum([ls_w_hat_n[ix_][ix2].energy() for ix_ in w_n_it]) / g_i_half
                      for ix2 in dup_it] #eq 11
        ls_S_hat_n=[ls_S_hat_sup[j].flatten()*(1.0/(ls_S_hat_n[j]+epsilon**2))
                    for j in dup_it] #eq 11
        S_hat_n_csr=su.flatten_list_to_csr(ls_S_hat_n)
        ls_w_hat_n=[(~A)*(w_n[ix_]*(tau**2)) for ix_ in w_n_it] #eq 21
        w_n_hat_flat=[su.flatten_list(ls_w_hat_n[ix_]) for ix_ in w_n_it]
        S_hat_bar_inv=su.inv_block_diag(tausq_AtA + (nu**2)*S_hat_n_csr,dict_in)
        w_n_hat_flat=[S_hat_bar_inv*w_n_hat_flat[ix_] for ix_ in w_n_it] #eq21 ctd
        ls_w_hat_n_unflat=[su.unflatten_list(w_n_hat_flat[ix_],A.duplicates) for ix_ in w_n_it]
        #the +0 in the next line returns a new WS object...
        ls_w_hat_n=[[ws_dummy.unflatten(w_hat_n_unflat)+0 for w_hat_n_unflat in ls_w_hat_n_unflat[ix_]]
                     for ix_ in w_n_it]
        w_bar_n=[A*ls_w_hat_n[ix_] for ix_ in w_n_it]  #eq 13
        w_bar_n=[ws_dummy.unflatten(w_bar_n[ix_])+0 for ix_ in w_n_it]#eq 13 contd
        #store back in the dict_in
        dict_in['ls_S_hat_n']=ls_S_hat_n
        dict_in['ls_w_hat_n']=ls_w_hat_n
        dict_in['w_bar_n']=w_bar_n

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

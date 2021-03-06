#!/usr/bin/python -tt
r"""This module implements the MSIST solver class component of
the py_solvers package.

The following mathematical definitions apply throughout:
:math:`\mathbf{Lambda}_{\alpha}`: The diagonal matrix of subband gain parameters.
:math:`\epsilon^2`: Continuation parameter which controls the convexity
    of the penalty surface.
:math:`\nu^2`: Continuation parameter which controls the sparse regularization.
:math:`\tau^2`: Cluster fidelity parameter (matrix) for MSIST-G and MSIST-C+G.
:math:`\mathbf{T}`: Cluster fidelity matrix for MSIST-G and MSIST-C+G.
:math:`\mathbf{A}`: Cluster averaging matrix.
:math:`\mathbf{G}`: Group energy averaging matrix.

"""

import numpy as np
from numpy import arange, conj, sqrt, median, abs as nabs, exp, maximum as nmax, angle, sum as nsum
from numpy.fft import fftn, ifftn
from numpy.linalg import norm
from scipy.sparse import csr_matrix, dia_matrix
import scipy.stats as ss
import os
import cPickle
import time

from py_utils.signal_utilities.ws import WS
import py_utils.signal_utilities.sig_utils as su
from py_utils.signal_utilities.sig_utils import unflatten_list as unflat_list
from py_operators.operator_ind import Operator
from py_operators.operator_comp import OperatorComp
from py_solvers.solver import Solver
from py_utils.section_factory import SectionFactory as sf

class MSIST(Solver):
    """Solver which implements the MSIST algorithm and its variants.

    Attributes:
        alpha (:class:`numpy.ndarray`): :math:`\mathbf{Lambda}_{\alpha}`.
        H (:class:`py_operators.operator_ind.Operator`): The forward modality.
        W (:class:`py_operators.operator_ind.Operator`): The forward transorm.

       spatial_threshold (:class:`bool`):
           :const:`True` if using a spatial hard threshold.
           :const:`False` (default) otherwise.
       spatial_threshold_val (:class:`float`): The spatial hard threhold value.
       input_complex (:class:`bool`):
           :const:`True` if signal to reconstruct is complex.
           :const:`False` (default) otherwise.
       input_phase_encoded (:class:`bool`): The signal to reconstruct is
           phase-encoded.
       input_poisson_corrupted (:class:`bool`):
           :const:`True` means observation is mixed Poisson-Gaussian (MSIST-P).
           :const:`False` (default) otherwise.
       sc_factor (:class:`float`): The factor :math:`F` (0,1] to use to adjust
           the scaling coefficients for wavelet domain variance estimation.
       convexnu (:class:`bool`):
           :const:`True` to use locally convex continuation rule.
           :const:`False` (default) otherwise.
       ordepsilon (:class:`bool`):
           :const:`True` to use order statistical :math:`\epsilon` continuation.
           :const:`False` (default) otherwise.
       ordepsilonpercstart (:class:`float`): Value of starting percentile for
           order statistical :math:`\epsilon` continuation.
       ordepsilonpercstop (:class:`float`): Value of stopping percentile for
           order statistical :math:`\epsilon` continuation.
       hmt (:class:`bool`):
           :const:`True` to use parent coefficients in coarse-to-fine precision
           estimation (MSIST only).
           :const:`False` (default) otherwise.
    """
    def __init__(self,ps_params,str_section):
        """ Class constructor for :class:`py_solvers.so_msist.MSIST`.

        """
        super(MSIST,self).__init__(ps_params,str_section)
        self.str_sparse_pen = self.get_val('sparsepenalty',False)
        self.alpha = None
        self.H = OperatorComp(ps_params,self.get_val('modalities',False))

        #avoid slow 'eval' in __mul__ of  these OperatorComp instances if
        #there is only one member.
        if len(self.H.ls_ops)==1:
            self.H = self.H.ls_ops[0]
        self.W = OperatorComp(ps_params,self.get_val('transforms',False))
        if len(self.W.ls_ops)==1:
            self.W = self.W.ls_ops[0]
        self.alpha = self.get_val('alpha',True)
        self.spatial_threshold = self.get_val('spatialthreshold',True) #msistp
        self.spatial_threshold_val = self.get_val('spatialthresholdval',True) #msistp
        self.input_complex = None
        self.input_phase_encoded = self.get_val('phaseencoded',True) #cplx
        self.input_poisson_corrupted = self.get_val('poissoncorrupted',True)
        self.sc_factor = self.get_val('scfactor',True,default_value=1)
        self.convexnu = self.get_val('convexnu',True) #if we have a constrained cont rule for nu
        self.ordepsilon = self.get_val('ordepsilon',True)
        self.hmt = self.get_val('hmt',True)
        self.ordepsilonpercstop = self.get_val('ordepsilonpercstop',True)
        self.ordepsilonpercstart = self.get_val('ordepsilonpercstart',True)
        self.profile = self.get_val('profile',True)

    def solve(self,dict_in):
        super(MSIST,self).solve()

        ##################################
        ### Transforms and Modalities ####
        ##################################
        H = self.H #mapping from solution domain to observation domain
        dict_in['H'] = H
        W = self.W #sparsifying transform
        dict_in['W'] = W
        # precision = 'float32'
        # if W.output_dtype!='':
        #     precision = W.output_dtype

        if self.alpha.__class__.__name__ != 'ndarray':
            self.alpha = su.spectral_radius(self.W, self.H, dict_in['x_0'].shape,
                                            self.get_val('alphamethod', False, 'spectrum'))
            # self.alpha = su.spectral_radius(self.W, self.H, (64,64,64),
            #                                 self.get_val('alphamethod', False, 'spectrum'))
        alpha = self.alpha #Lambda_alpha main diagonal (B-sized vector of subband gains)
        dict_in['alpha'] = alpha
        ############
        #Input Data#
        ############

        if H.output_fourier:
            y_hat = dict_in['y']
        else:
            #do an extra FFT to do deconvolution in fourier domain
            y_hat = fftn(dict_in['y'])

        x_n = dict_in['x_0'].copy() #seed current solution
        # The famous Joan Lasenby "residuals"
        dict_in['x_n'] = x_n
        x = dict_in['x'].copy()
        dict_in['resid_n'] = x - x_n
        x_max = np.max(x)
        x_min = np.min(x)
        # dict_in['resid_range'] = np.array([x_min - x_max, x_max + x_max])
        dict_in['resid_range'] = np.array([-255.0/2, 255.0/2])
        #######################
        #Common Initialization#
        #######################

        #determine whether/not we need double the wavelet transforms on
        #each iteration for a complex-valued input signal
        self.input_complex = np.iscomplexobj(x_n)

        #initialize current solution in sparse domain
        #g_i is the element group size (2 for CWT, 4 for CWT and input_complex)
        if self.input_complex:
            if self.input_phase_encoded:
                theta_n = su.phase_unwrap(angle(x_n),
                                          dict_in['dict_global_lims'],
                                          dict_in['ls_local_lim_secs'])
            else:
                theta_n = angle(x_n)
            dict_in['theta_n'] = theta_n
            dict_in['magnitude_n'] = nabs(x_n)

            w_n = [W * x_n.real, W * x_n.imag]
            g_i = 2 * (w_n[0].is_wavelet_complex() + 1)
        else:
            w_n = [W * x_n]
            g_i = (w_n[0].is_wavelet_complex() + 1)
        w_n_len = len(w_n)
        w_n_it = xrange(w_n_len) #iterator for w_n
        dict_in['w_n'] = w_n

        #initialize the precision matrix with zeros
        S_n = w_n[0]*0
        dict_in['S_n'] = S_n
        #initialize continuation parameters
        epsilon,nu = self.get_epsilon_nu()
        if self.ordepsilon:
            # self.ordepsilonpercstart = 8.0/9.0*(.55**2)
            epsilon = np.zeros(self.int_iterations+1,)
            self.percentiles = np.arange(30,self.ordepsilonpercstop,-1.0/self.int_iterations)
            epsilon[0] = self.get_ord_epsilon(w_n[0],np.inf, self.percentiles[0])
            dict_in['epsilon_sq'] = epsilon[0]**2
        else:
            dict_in['epsilon_sq'] = epsilon**2
        if self.convexnu:
            nu = np.zeros(self.int_iterations+1,)
            nu[0] = self.get_convex_nu(w_n[0],
                                       epsilon[0]**2,
                                       np.min(alpha))
            dict_in['nu_sq'] =  nu[0]**2
        else:
            dict_in['nu_sq'] = nu**2

        #wavelet domain variance used for poisson deblurring
        ary_p_var=0

        ########################################
        #Sparse penalty-specific initialization#
        ########################################

        if self.str_sparse_pen == 'l0rl2_bivar':
            w_tilde = w_n[0]*0
            sqrt3=sqrt(3.0)
            sigsq_n = self.get_val('nustop', True)**2
            sig_n = sqrt(sigsq_n)

        if self.str_sparse_pen == 'l0rl2_group':
            tau = self.get_val('tau',True)
            tau_rate = self.get_val('taurate',True)
            tau_start = self.get_val('taustart',True)
            if np.all(tau_start != 0) and tau_rate != 0:
                tau_end = tau
                tau = tau_start
            A = sf.create_section(self.ps_parameters,
                                  self.get_val('clusteraverage',False)) #cluster
            G = sf.create_section(self.ps_parameters,
                                  self.get_val('groupaverage',False)) #group
            #initialize A and G with parameters of the master vector
            A.init_csr_avg(w_n[0])
            G.init_csr_avg(w_n[0])
            dup_it = xrange(A.duplicates) # iterator for duplicate variable space

            #initialize non-overlapping space (list of ws objects ls_w_hat_n)
            ls_w_hat_n = [[w_n[ix_]*1 for j in dup_it] for ix_ in w_n_it]

            #initialize non-overlapping space precision
            ls_S_hat_n = [((sum([w_n[ix].energy() for ix in w_n_it]) / g_i)
                           + epsilon[0]**2).invert() for int_dup in dup_it]
            w_bar_n = [w_n[ix_]*1 for ix_ in w_n_it]

            #using the structure of A, initialize the support of Shat, what
            A_row_ix=np.nonzero(A.csr_avg)[0]
            A_col_ix=np.nonzero(A.csr_avg)[1]
            D=csr_matrix((np.ones(A_col_ix.size,),(A_row_ix,A_col_ix)),
                         shape=A.csr_avg.shape)

            #compute the support of Shat
            ls_S_hat_sup=unflat_list(D.transpose()*((w_n[0]*0+1).flatten()),
                                     A.duplicates)

            #load this vector into each new wavelet subband object
            ls_S_hat_sup = [(w_n[0]*0).unflatten(S_sup) for S_sup in ls_S_hat_sup]
            ls_S_hat_sup = [S_hat_n_sup.nonzero() for S_hat_n_sup in ls_S_hat_sup]
            del S_sup
            del S_hat_n_sup
            #precompute AtA (doesn't change from one iteration to the next)
            AtA = (A.csr_avg.transpose()*A.csr_avg).tocsr()

            #convert tau**2 to csr format to allow for subband-adaptive constraint
            if tau.__class__.__name__ != 'ndarray':
                tau_sq = np.ones(w_n[0].int_subbands) * tau**2
            else:
                tau_sq = tau**2
            tau_sq_dia = [((w_n[0]*0+1).cast(A.dtype))*tau_sq for j in dup_it]
            # tau_sq_dia = [((w_n[0]*0+1))*tau_sq for j in dup_it]
            tau_sq_dia = su.flatten_list(tau_sq_dia)
            offsets = np.array([0])
            tau_sz = tau_sq_dia.size
            tau_sq_dia = dia_matrix((tau_sq_dia, offsets), shape=(tau_sz, tau_sz))


            #initialize S_hat_bar parameters for efficient matrix inverses
            Shatbar_p_filename = A.file_path.split('.pkl')[0]+'Shatbar.pkl'
            if not os.path.isfile(Shatbar_p_filename):
                dict_in['col_offset']=A.int_size
                S_hat_n_csr=su.flatten_list_to_csr(ls_S_hat_sup)
                su.inv_block_diag((tau_sq_dia) * AtA + S_hat_n_csr, dict_in)
                filehandler=open(Shatbar_p_filename, 'wb')
                cPickle.dump(dict_in['dict_bdiag'],filehandler,-1)
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
            dict_in['AtA']=AtA
            dict_in['ls_S_hat_sup']=ls_S_hat_sup
            dict_in['w_n_it']=w_n_it
            dict_in['dup_it']=dup_it
            dict_in['ws_dummy']=w_n[0]*0
            dict_in['g_i']=g_i

            # self.update_duplicates(dict_in,nu[0],epsilon[0],tau_sq, tau_sq_dia)

            w_bar_n=dict_in['w_bar_n']
            ls_w_hat_n=dict_in['ls_w_hat_n']
            ls_S_hat_n=dict_in['ls_S_hat_n']
            del D #iterations need A and G only, not D

        if (self.str_sparse_pen == 'vbmm' or  #vbmm
            self.str_sparse_pen == 'vbmm_hmt'):
            p_a = self.get_val('p_a',True)
            p_b_0 = self.get_val('p_b_0',True)
            p_k = self.get_val('p_k',True)
            p_theta = self.get_val('p_theta',True)
            p_c = self.get_val('p_c',True)
            p_d = self.get_val('p_d',True)
            b_n = w_n[0]*0
            sigma_n = 0
            if self.str_sparse_pen == 'vbmm_hmt':
                ary_a = self.get_gamma_shapes(W * dict_in['x_0'])
                b_n = w_n[0] * p_b_0

        #poisson + gaussiang noise,
        #using the scaling coefficients in the regularization (MSIST-P)
        if self.input_poisson_corrupted:
            #need a 0-padded y to get the right size for the scaling coefficients
            b = dict_in['b']
            if not H.output_fourier:
                y_hat = fftn(dict_in['y'] - b)
            else:
                y_hat = fftn(ifftn(dict_in['y']) - b)
            w_y = (W*dict_in['y_padded'])
            dict_in['x_n']=su.crop_center(x_n,dict_in['y'].shape)
            w_y_scaling_coeffs=w_y.downsample_scaling()

        self.results.update(dict_in)
        print 'Finished itn: n=' + str(0)
        #begin iterations here for the MSIST(-X) algorithm, add some profiling info here
        if self.profile:
            dict_profile={}
            dict_profile['twoft_time']=[]
            dict_profile['wht_time']=[]
            dict_profile['other_time']=[]
            dict_profile['reproj_time_inv']=[]
            dict_profile['reproj_time_for']=[]
            dict_in['profiling']=dict_profile
            t0=time.time()
        ####################
        ##Begin Iterations##
        ####################
        for n in np.arange(self.int_iterations):
            ####################
            ###Landweber Step###
            ####################
            twoft_0=time.time()
            H.set_output_fourier(True) #force Fourier output to reduce ffts
            if self.input_complex:
                f_resid = y_hat - H * x_n #Landweber difference
            else:
                f_resid = ifftn(y_hat - H * x_n)
                H.set_output_fourier(False)
            twoft_1=time.time()
            if self.input_complex:
                HtHf = (~H) * f_resid
                w_resid = [W * (HtHf).real, W * (HtHf).imag]
            else:
                w_resid = [W * ((~H) * f_resid)]
            wht=time.time()
            if self.profile:
                dict_profile['twoft_time'].append(twoft_1-twoft_0)
                dict_profile['wht_time'].append(wht-twoft_1)
            ########################
            ######Convex Nu#########
            #####Ord/HMT Epsilon####
            ########################
            if self.ordepsilon:
                if n==0:
                    prevepsilon = epsilon[0]
                else:
                    prevepsilon = epsilon[n-1]
                epsilon[n] = self.get_ord_epsilon(w_n[0], prevepsilon, self.percentiles[n])
                dict_in['epsilon_sq'] = epsilon[n] ** 2
            if self.convexnu:
                nu[n] = self.get_convex_nu(w_n[0], epsilon[n]**2, np.min(self.alpha))
                dict_in['nu_sq'] = nu[n] ** 2

            ###############################################
            ###Sparse Penalty-Specific Thresholding Step###
            ###############################################
            if self.str_sparse_pen == 'l0rl2_group':
                #S_hat_n, w_hat_n, and wb_bar (eqs 11, 19, and 13)
                self.update_duplicates(dict_in, nu[n], epsilon[n], tau_sq, tau_sq_dia)
                w_bar_n=dict_in['w_bar_n']
                ls_w_hat_n=dict_in['ls_w_hat_n']

            #####################################################
            #Subband-adaptive subband update of precision matrix#
            #####################################################
            if (self.str_sparse_pen[0:5] == 'l0rl2' and
                self.str_sparse_pen[-5:] != 'bivar'):
                if self.str_sparse_pen == 'l0rl2_group':
                    S0_n = nsum([nabs(w_n[ix].ary_lowpass)**2 for ix in w_n_it],
                                axis=0) / g_i  + epsilon[n]**2
                    S0_n = 1.0 / S0_n
                else:
                    if self.hmt:
                        S_n_prev = S_n*1.0
                        S_n.set_subband(0,(1.0 /
                                           ((1.0 / g_i) *
                                            nabs(w_n[0].get_subband(0))**2 + (epsilon[n]**2))))

                        for s in xrange(w_n[0].int_subbands-1,0,-1):
                            sigma_sq_parent_us = nabs(w_n[0].get_upsampled_parent(s))**2
                            s_parent_sq = 1.0/((2.0**(-2.25))*(1.0/g_i*sigma_sq_parent_us))
                            S_n.set_subband(s,s_parent_sq)

                    else:
                        S_n = (sum([w_n[ix_].energy() for ix_ in w_n_it]) / g_i
                               + epsilon[n]**2).invert()
            elif (self.str_sparse_pen[0:5] == 'vbmm' and
                  self.str_sparse_pen[-5:] != 'hmt'):
                cplx_norm = 1.0 + self.input_complex
                S_n = ((g_i + 2.0 * p_a) *
                       (sum([w_n[ix_].energy() for ix_ in w_n_it]) / cplx_norm
                       + sigma_n + 2.0 * b_n).invert())
                b_n = (p_k + p_a) * (S_n.get_subband(s) + p_theta).invert()
                sigma_n = (1.0 / nu[n]**2 * alpha[s] + S_n).invert()

            else:
                #iterating through subbands is necessary, coarse to fine
                for s in xrange(w_n[0].int_subbands-1,-1,-1):
                    #Sendur Selesnick BSWLVE paper
                    if self.str_sparse_pen == 'l0rl2_bivar':
                        if s > 0:
                            s_parent_us = nabs(w_n[0].get_upsampled_parent(s))**2
                            s_child = nabs(w_n[0].get_subband(s))**2
                            yi,yi_mask = su.get_neighborhoods(s_child,1) #eq 8
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

                    elif self.str_sparse_pen == 'vbmm_hmt': #vbmm
                        if n == 0:
                            sigma_n = 0
                        else:
                            sigma_n = (1.0 / nu[n]**2 * alpha[s] + S_n.get_subband(s))**(-1)
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
                            S_n.set_subband(s, (g_i + 2.0 *  ary_a[s]) /
                                            (nabs(w_n[0].get_subband(s))**2 + sigma_n +
                                             2.0 * b_n.get_subband(s)))
                            b_n.set_subband(s,ary_a[s] * w_en_avg)
                        else: #no parents, so generate fixed-param gammas
                            S_n.set_subband(s, (g_i + 2.0 * ary_a[s]) /
                                            (nabs(w_n[0].get_subband(s))**2 + sigma_n +
                                             2.0 * b_n.get_subband(s)))
                            b_n.set_subband(s, (p_k + ary_a[s]) /
                                            (S_n.get_subband(s) + p_theta))
                    else:
                        raise ValueError('no such solver variant')
            #########################
            #Update current solution#
            #########################
            for s in xrange(w_n[0].int_subbands-1,-1,-1):
                if self.input_poisson_corrupted:
                    if s==0:
                        ary_p_var = w_y.ary_lowpass
                    else:
                        int_lev, int_ori = w_n[0].lev_ori_from_subband(s)
                        ary_p_var = w_y_scaling_coeffs[int_lev]
                        ary_p_var[ary_p_var<=0]=0
                if (self.str_sparse_pen == 'l0rl2_group'):
                    if s>0:
                        for ix_ in w_n_it:
                            w_n[ix_].set_subband(s,
                                            (alpha[s] * w_n[ix_].get_subband(s) + w_resid[ix_].get_subband(s) +
                                             (tau_sq[s])*w_bar_n[ix_].get_subband(s)) / (alpha[s] + tau_sq[s]))
                    else: #a standard msist update for the lowpass coeffs
                        for ix_ in w_n_it:
                            w_n[ix_].set_subband(s, \
                                                 (alpha[s] * w_n[ix_].get_subband(s) + w_resid[ix_].get_subband(s)) /
                                                 (alpha[s] + (nu[n]**2) * S0_n))
                else:
                    for ix_ in w_n_it:
                        w_n[ix_].set_subband(s,
                                           (alpha[s] * w_n[ix_].get_subband(s) + w_resid[ix_].get_subband(s)) /
                                           (alpha[s] + (nu[n]**2+self.sc_factor*ary_p_var) * S_n.get_subband(s)))
                #end updating subbands

            #############################################
            ##Solution Domain Projection and Operations##
            #############################################
            tother = time.time()
            if self.input_complex:
                x_n = np.asfarray(~W * w_n[0],'complex128')
                x_n += 1j*np.asfarray(~W * w_n[1],'complex128')
                m_n = nabs(x_n)
                theta_n = angle(x_n)
                if self.input_phase_encoded: #need to apply boundary conditions for phase encoded velocity
                    #the following isn't part of the documented algorithm
                    #it only needs to be executed at the end to fix
                    #phase wrapping in very high dynamic-phase regions
                    theta_n = su.phase_unwrap(angle(x_n),
                                              dict_in['dict_global_lims'],
                                              dict_in['ls_local_lim_secs'])
                    if self.get_val('iterationmask',True): #apply boundary conditions for phase encoded velocity
                        theta_n *= dict_in['mask']
                        if self.get_val('magnitudemask',True,1):
                            m_n *= dict_in['mask'] #uncomment this for 'total' masking
                    x_n = m_n * exp(1j*theta_n)
                dict_in['theta_n'] = theta_n
                dict_in['magnitude_n'] = m_n
            else:
                x_n = ~W * w_n[0]
            tinvdwt = time.time()
            #implicit convolution operator is used, so crop and repad
            if H.str_object_name=='Blur' and H.lgc_even_fft:
                x_n=su.crop_center(x_n,dict_in['y'].shape)
            if self.input_poisson_corrupted and self.spatial_threshold:
                x_n[x_n<self.spatial_threshold_val]=0.0

            #finished spatial domain operations on this iteration, store
            dict_in['x_n'] = x_n
            # store "residuals"
            dict_in['resid_n'] = x - x_n
            # dict_in['resid_range'] = np.array([np.min(dict_in['resid_n']), np.max(dict_in['resid_n'])])
            print "resid min " + str(np.round(np.min(dict_in['resid_n']), 2))
            print "resid max " + str(np.round(np.max(dict_in['resid_n']), 2))

            if H.str_object_name=='Blur' and H.lgc_even_fft:
                x_n=su.pad_center(x_n, dict_in['x_0'].shape)

            #############################
            #Wavelet Domain Reprojection#
            #############################
            if self.profile:
                dict_profile['other_time'].append(tother-wht)
            if self.input_complex:
                w_n = [W * x_n.real, W * x_n.imag]
            else:
                w_n = [W * x_n]
            tforwardwt = time.time()
            if self.profile:
                dict_profile['reproj_time_inv'].append(tinvdwt-tother)
                dict_profile['reproj_time_for'].append(tforwardwt-tinvdwt)
            if self.str_sparse_pen[:11] == 'l0rl2_group':
                ls_w_hat_n = [[ls_w_hat_n[ix_][j] * ls_S_hat_sup[j] +
                               w_bar_n[ix_] * ((ls_S_hat_sup[j]+(-1))*(-1))
                               for j in dup_it] for ix_ in w_n_it] #fill in the gaps with w_bar_n
                w_bar_n = [W*((~W)*w_bar_n[ix_]) for ix_ in w_n_it]
                ls_w_hat_n = [[W*((~W)*w_hat_n) for w_hat_n in ls_w_hat_n[ix_]]
                              for ix_ in w_n_it]
                dict_in['w_bar_n'] = w_bar_n
                dict_in['ls_w_hat_n'] = ls_w_hat_n
                if tau_rate != 0 and not np.any(tau > tau_end):
                    tau_sq_dia = tau_rate * tau_sq_dia
                    tau = np.sqrt(tau_rate) * tau
            dict_in['w_n'] = w_n
            dict_in['S_n'] = S_n
            ################
            #Update Results#
            ################
            self.results.update(dict_in)
            print 'Finished itn: n=' + str(n+1)
            # if self.str_sparse_pen[:11] == 'l0rl2_group' and n==150: #an interesting experiment for cs..
            #     self.str_sparse_pen = 'l0rl2'

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
                                  for i in arange(self.int_iterations+1)])
            nu = np.asarray([nmax(nu_start * (decay ** i),nu_stop) \
                                  for i in arange(self.int_iterations+1)])
        elif str_method == 'exponential':
            epsilon = np.asarray([epsilon_start * exp(-i / decay) + epsilon_stop \
                                  for i in arange(self.int_iterations+1)])
            nu = np.asarray([nu_start * exp(-i / decay) + nu_stop \
                                  for i in arange(self.int_iterations+1)])
        elif str_method == 'fixed':
            epsilon = epsilon_start*np.ones(self.int_iterations+1,)
            nu = nu_start*np.ones(self.int_iterations+1,)
        else:
            raise Exception('no such continuation parameter rule')
        return epsilon,nu

    def update_duplicates(self, dict_in, nu, epsilon, tau_sq, tau_sq_dia):
        '''update the duplicate estimates in (the msist-g algorithm updates)
        ls_S_hat_n
        ls_w_hat_n
        w_bar_n

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
        AtA=dict_in['AtA']
        ls_S_hat_sup=dict_in['ls_S_hat_sup']
        g_i = dict_in['g_i']
        w_n_it = dict_in['w_n_it']
        dup_it = dict_in['dup_it']
        #perform the updats
        ls_S_hat_n=G*[sum([ls_w_hat_n[ix_][ix2].energy() for ix_ in w_n_it]) / g_i
                      for ix2 in dup_it] #eq 11
        ls_S_hat_n=[ls_S_hat_sup[j].flatten()*(1.0/(ls_S_hat_n[j]+epsilon**2))
                    for j in dup_it] #eq 11
        S_hat_n_csr=su.flatten_list_to_csr(ls_S_hat_n)
        ls_w_hat_n=[(~A)*(w_n[ix_]*(tau_sq)) for ix_ in w_n_it] #eq 21
        w_n_hat_flat=[su.flatten_list(ls_w_hat_n[ix_]) for ix_ in w_n_it]
        S_hat_bar_inv=su.inv_block_diag((tau_sq_dia) * AtA + (nu**2) * S_hat_n_csr, dict_in)
        w_n_hat_flat=[S_hat_bar_inv*w_n_hat_flat[ix_] for ix_ in w_n_it] #eq21 ctd
        ls_w_hat_n_unflat=[unflat_list(w_n_hat_flat[ix_],A.duplicates) for ix_ in w_n_it]
        #the +0 in the next line returns a new WS object...
        ls_w_hat_n=[[ws_dummy.unflatten(w_hat_n_unflat)+0 for w_hat_n_unflat in ls_w_hat_n_unflat[ix_]]
                     for ix_ in w_n_it]
        w_bar_n=[A*ls_w_hat_n[ix_] for ix_ in w_n_it]  #eq 13
        w_bar_n=[ws_dummy.unflatten(w_bar_n[ix_])+0 for ix_ in w_n_it]#eq 13 contd
        #store back in the dict_in
        dict_in['ls_S_hat_n']=ls_S_hat_n
        dict_in['ls_w_hat_n']=ls_w_hat_n
        dict_in['w_bar_n']=w_bar_n

    def get_gamma_shapes(self, ws_coeffs):
        r"""Use maximum likelihood to obtain the shape parameters for a
        Gauss-Gamma (Student-T) distribution of wavelet coefficients.

        Args:
            ws_coeffs (:class:`py_utils.signal_utilities.WS`): The
                wavelet coefficents.

        .. note:: Experimental feature to test improvements over
            using single shape parameter. Works for 2D signals only.

        Returns:
            :class:`numpy.ndarray`: A 1-d array of shape parameters.
        """
        ary_a = np.zeros(ws_coeffs.int_subbands,)
        for s in xrange(1,ws_coeffs.int_subbands,ws_coeffs.int_orientations):
            for s2 in xrange(s,s+ws_coeffs.int_orientations):
                subband = ws_coeffs.get_subband(s2).flatten()
                subband_ri = np.concatenate((subband.real,subband.imag))
                if s2==s:
                    subband_tot=np.zeros(subband_ri.size*4)
                    tot_45=np.zeros(subband_ri.size*2)
                    index=0
                    index_45=0
                if np.mod(s2-2,6)==0 or np.mod(s2-5,6)==0:
                    tot_45[index_45:index_45+subband_ri.size]=subband_ri.copy()
                    index_45=index_45+subband_ri.size
                else:
                    subband_tot[index:index+subband_ri.size]=subband_ri.copy()
                    index=index+subband_ri.size
            fit_nu, fit_loc, fit_scale = ss.t.fit(subband_tot)
            ary_a[s] = fit_nu / 2
            ary_a[s+2:s+4] = fit_nu / 2
            ary_a[s+5] = fit_nu / 2
            fit_nu, fit_loc, fit_scale = ss.t.fit(tot_45)
            ary_a[s+1] = fit_nu / 2
            ary_a[s+4] = fit_nu / 2
        return ary_a

    def get_convex_nu(self, w_n, epsilon_sq, alpha_min):
        r"""Find the minimum guaranteed locally convex :math:`nu`.

        Find the minimum guaranteed locally convex :math:`nu`
        using the minimum entry of
        :math:`\mathbf{Lambda}_{\alpha}`, :math:`{\alpha}_{\min}`,
        the value of :math:`\epsilon^2`, and
        :math:`\mathbf{w}_{n}` according to the following formula:

        :math:`\nu^2=\frac{\alpha_{_min}}{8}
        \min_{m\in\{1\ldots M\}}(w_m^2+4\epsilon^2+\frac{4\epsilon^2}{w_m^2})`

        Args:
            w_n (:class:`py_utils.signal_utilities.WS`): The
                wavelet coefficents.
            epsilon_sq (:class:`float`): The value of :math:`\epsilon^2`.
            alpha_min (:class:`float`): The minimum value
                of :math:`\mathbf{Lambda}_{\alpha}`.

        .. note:: Experimental feature to test automatic convex continuation.

        Returns:
            :class:`float`: The value of :math:`\nu`.
        """

        rhsmin = w_n.energy().flatten()
        rhsmin = rhsmin + 4*epsilon_sq + 4*epsilon_sq**2/rhsmin
        rhsmin = np.min(rhsmin)
        lhsmin = alpha_min
        return np.sqrt(1.0/8*lhsmin*rhsmin)

    def get_ord_epsilon(self, w_n, epsilon_max, percetile_n):
        r"""Find the value of :math:`\epsilon` according to
        reference percentile in a vector of wavelet coefficienets.

        Args:
            w_n (:class:`py_utils.signal_utilities.WS`): The reference
                wavelet coefficents.
            epsilon_max (:class:`float`): Max return value of :math:`\epsilon`.
            percetile_n (:class:`float`): Percentile (0,1) in to return in
                reference wavelet coefficients.

        .. note:: Experimental feature to test automatic percentile-based
            continuation.

        Returns:
            :class:`float`: The value of :math:`\epsilon`.
        """

        coeffs = w_n.energy().flatten()
        return min(epsilon_max, np.sqrt(np.percentile(coeffs,percetile_n)))

    class Factory:
        def create(self,ps_params,str_section):
            return MSIST(ps_params,str_section)

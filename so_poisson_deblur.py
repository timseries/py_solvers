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
from scipy.optimize import fmin_ncg as ncg
from numpy.linalg import norm

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
        sigma = dict_in['noisevariance']
        dict_in['x_n']  = x_n
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
        for n in np.arange(self.int_iterations):
            #update S
            for s in arange(1,w_n.int_subbands):
                S_n.set_subband(s,(1.0 / ((1.0 / g_i) * nabs(w_n.get_subband(s))**2 + epsilon[n]**2)))
                S_n.flatten(thresh=True)
            #update q
            q_n = (dict_in['y'] / sigma + self.u(w_n,b)) / (1/sigma + 1)
            #update w
            w_n.flatten()
            w_n.ws_vector = ncg(self.F,w_n.ws_vector,self.F_prime,None,args=(w_n,q_n,b,S_n.ws_vector), maxiter=5)
            w_n.unflatten()
            x_n = ~W * w_n
            w_n = W * x_n #reprojection, to put our iterate in the range space, prevent drifting
            dict_in['x_n'] = x_n
            dict_in['w_n'] = w_n

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
            epsilon = np.ones(self.int_iterations)*epsilon_start
            nu = np.ones(self.int_iterations)*nu_start
        else:
            raise Exception('no such continuation parameter rule')
        return epsilon,nu
            
    def F(self, w, *args):
        '''
        The function of the wavelet coefficients we wish to minimze in order to iterate our solution 
        using Newton/Secant/Halley's method. w is a ndarray
        '''
        ws = args[0]
        q =  args[1]
        b =  args[2]
        S =  args[3]
        ws.ws_vector = w
        ws.unflatten()
        u = self.u(ws,b)
        print 'ndim u: ' + str(u.ndim)
        print 'ndim q: ' + str(q.ndim)
        w_threshold = norm(u.flatten()-q.flatten(),ord=2)**2
        w_threshold = w_threshold + np.dot(S,w**2)
        #for s in arange(1,w_threshold.int_subbands):
        #    w_threshold.set_subband(w_threshold.get_subband(s) + S.get_subband(s))

        return w_threshold / 2.0
    
    def F_prime(self, w, *args):
        '''
        The function of the wavelet coefficients we wish to minimze in order to iterate our solution 
        using Newton/Secant/Halley's method. w is a ndarray
        '''
        ws = args[0]
        q =  args[1]
        b =  args[2]
        S =  args[3]
        ws.ws_vector = w
        ws.unflatten()
        u = self.u(ws,b)
        u_prime = self.u_prime(ws,b)
        w_threshold = (self.W * ifftn(~self.H * (u_prime * (u - q))))
        w_threshold.flatten()
        w_threshold = w_threshold.ws_vector + S*w
        #for s in arange(1,w_threshold.int_subbands):
        #    w_threshold.set_subband(w_threshold.get_subband(s) + S.get_subband(s))

        return w_threshold / 2.0

    def F_prime_old(self, w, *args):
        '''
        Derivative of F
        '''
        ws = args[0]
        q = args[1]
        b = args[2] 
        S = args[3]
        ws.ws_vector = w
        ws.unflatten()
        u_prime = self.u_prime(ws,b)
        u_prime_prime = self.u_prime_prime(ws,b)
        u = self.u(ws,b)
        w_threshold = self.W * ifftn(~self.H * (u_prime**2 + u_prime_prime*(u-q)))
        w_threshold.flatten()
        w_vector = w_threshold.ws_vector + S
        #for s in arange(1,w_threshold.int_subbands):
        #    w_threshold.set_subband(w_threshold.get_subband(s)+S.get_subband(s))
        return 1 / 2.0 * w_vector
    
    def f(self, ws, b):
        '''
        Returns the Anscombe transformation of the linear blurring equation 2*sqrt(AW^Tw+b)
        '''
        b1 = b + 3 / 8.0      
        return 2*sqrt(ifftn(self.H * (~self.W * ws)) + b1)

    def f_prime(self, ws, b):
        '''
        The derivative of f(.)
        '''
        return 1 / (2 * self.f(ws,b))

    def u(self, ws, b):
        ''' 
        Returns the mean of the Anscombe-transformed Poisson random variable
        '''
        f = self.f(ws,b)
        return f - 1 / (2 * f)

    def u_prime(self, ws, b): 
        ''' 
        Returns the derivative of the mean of the Anscombe-transformed Poisson random variable
        '''
        f = self.f(ws,b)
        return (2 / f + 1 / (f**3))

    def u_prime_prime(self, ws, b):
        ''' 
        Derivative of u_prime
        '''
        f = self.f(ws,b)
        return -4 / f**3 - 6/f**5

    def old_school_solver():        
        for n in np.arange(self.int_iterations):
            f_2 = ifftn(H * x_n).real + b
            f_1 = f_2 + 3/8.0
            print 'f1 and f2 min...'
            print f_1.min()
            print f_2.min()
            print 'f1 and f2 max...'
            print f_1.max()
            print f_2.max()
            u = 2 * sqrt(f_1) - 1 / (4 * sqrt(f_2))
            u_sq = u**2
            print 'u_sq max...'
            print u_sq.max()

            u_sq_f_1 = sqrt(np.sum(u_sq * f_1))
            u_sq_f_2 = sqrt(np.sum(u_sq * f_2))
            print 'u_sq_f_1 max...'
            print u_sq_f_1

            f_resid = (-u_sq/u_sq_f_1 + u_sq_f_2/(4*f_2**2) \
              - u_sq/(8*f_2*u_sq_f_2) + 4*ones(u_sq_f_2.shape) \
              - 1/(16*f_2))
            print 'f_resid max/min...'
            print f_resid.max()
            print f_resid.min()
            w_n = ifftn(~H * f_resid).real
            print 'w_n max/min...'
            print w_n.max()
            print w_n.min()
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
            
    class Factory:
        def create(self,ps_parameters,str_section):
            return PoissonDeblur(ps_parameters,str_section)

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

from py_utils.signal_utilities.sig_utils import crop_center

#profiling and debugging stuff
from py_utils.timer import Timer
import pdb
import time

class RichardsonLucy(Solver):
    """
    Solver which performs the Poisson Deblurring Algorithm (for widefield microscopy deconvolution)
    """
    def __init__(self,ps_parameters,str_section):
        """
        Class constructor for DTCWT
        """
        super(RichardsonLucy,self).__init__(ps_parameters,str_section)
        self.str_solver_variant = self.get_val('solvervariant',False)
        self.H = OperatorComp(ps_parameters,self.get_val('modalities',False))
        self.H = self.H.ls_ops[0] #assume we just have one transform
        self.str_group_structure = self.get_val('grouptypes',False)
        
            
    def solve(self,dict_in):
        """
        Takes an input object (ground truth, forward model observation, metrics required)
        Returns a solution object based on the solver this object was instantiated with.
        """
        super(RichardsonLucy,self).solve()
        H = self.H
        #input data 
        x_n = dict_in['x_0'].copy()
        b = dict_in['b']#background
        sigma_sq = dict_in['noisevariance']
        #dummy multiply to intialize H
        H*x_n
        dict_in['x_n'] = su.crop_center(x_n,dict_in['y'].shape)
        gamma = (~H)*np.ones(dict_in['y'].shape)
        #begin iterations here
        self.results.update(dict_in)
        print 'Finished itn: n=' + str(0)
        dict_profile={}
        dict_profile['twoft_time']=[]
        dict_profile['other_time']=[]
        dict_profile['ht_time']=[]
        dict_in['profiling']=dict_profile
        for n in np.arange(self.int_iterations):
            #save current iterate
            twoft_0=time.time()
            div=(H*x_n+b)
            twoft_1=time.time()
            dict_profile['twoft_time'].append(twoft_1-twoft_0)            
            other_time_0=time.time()
            div = dict_in['y']/div
            div[div==np.nan]=0.0    
            other_time_1=time.time()
            dict_profile['other_time'].append(other_time_1-other_time_0)
            twoft_2=time.time()
            x_n = ((~H) * div) * x_n / gamma
            twoft_3=time.time()
            dict_profile['ht_time'].append(twoft_3-twoft_2)
            x_n=su.crop_center(x_n,dict_in['y'].shape)
            dict_in['x_n'] = x_n
            x_n=su.pad_center(x_n,dict_in['x_0'].shape)
            #update results
            self.results.update(dict_in)
            print 'Finished itn: n=' + str(n+1)
        return dict_in

    class Factory:
        def create(self,ps_parameters,str_section):
            return RichardsonLucy(ps_parameters,str_section)

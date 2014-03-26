#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj, sqrt,median, abs as nabs, exp, maximum as nmax
from numpy.fft import fftn, ifftn
from numpy.linalg import norm

from sklearn import svm

from py_utils.signal_utilities.ws import WS
from py_utils.signal_utilities.scat import Scat
import py_utils.signal_utilities.sig_utils as su
from py_solvers.solver import Solver
from py_operators.operator import Operator
from py_operators.operator_comp import OperatorComp
from py_utils.section_factory import SectionFactory as sf

import scipy.stats as ss

class Classify(Solver):
    """
    General classify class which performs the feature computation, training, and validation.
    """
    def __init__(self,ps_params,str_section):
        """
        Class constructor for DTCWT
        """
        super(Classify,self).__init__(ps_params,str_section)
        self.S = OperatorComp(ps_params,
                              self.get_val('transforms',False))
        if len(self.S.ls_ops)==1: #avoid slow 'eval' in OperatorComp
            self.S = self.S.ls_ops[0] 
        self.classifier_method = self.get_val('method',False)
        self.feature_sec_in = self.get_val('featuresectioninput',False)
        self.feature_sec_out = self.get_val('featuresectionoutput',False)
        if self.classifier_method=='svc':
            self.clf = svm.SVC()
        elif self.classifier_method=='linearsvc':
            self.clf = svm.LinearSVC()
        else:
            ValueError('unknown classification method ' + self.classifier_method)    
        
    def solve(self,dict_in):
        """
        Takes an input object (ground truth, forward model observation, metrics required)
        Returns a solution object based on the solver this object was instantiated with.
        """
        super(Classify,self).solve()
        S = self.S
        feature_reduce = Scat().reduce #function handle 
        classes = dict_in['x'].keys()
        if self.feature_sec_in != '':
            sec_input = sf.create_section(self.get_params(),self.feature_sec_in)
            dict_in['x_feature'] = sec_input.read(dict_in,return_val=True)
        else:   
            met_output_obj = sf.create_section(self.get_params(),self.feature_sec_out)
            #generate the features if not previously stored from a prior run,takes a while...
            #TODO:parallelize
            for _class_index,_class in enumerate(classes):
                print 'generating scattering features for class ' + _class
                n_samples = len(dict_in['x'][_class])
                dict_in['x_feature'][_class]=[feature_reduce((S*dict_in['x'][_class][sample]).flatten()) for 
                                              sample in xrange(n_samples)]
            #update and save
            met_output_obj.update(dict_in)
            self.results.save_metric(met_output_obj)

        #assumes each feature vector is the same size    
        dict_in['feature_vector_size'] = dict_in['x_feature'][classes[0]][-1].size
        Xtrain = np.zeros([dict_in['n_training_samples'],dict_in['feature_vector_size']]) 
        Xtest = np.zeros([dict_in['n_testing_samples'],dict_in['feature_vector_size']]) 
        
        #vector to hold the training labels
        ytrain = np.zeros(dict_in['n_training_samples'],)
        ytest = np.zeros(dict_in['n_testing_samples'],)
        sample_index = 0

        #generate the training and test data matrix (X) and label vectors (y)
        for _class_index,_class in enumerate(classes):
            for training_index in dict_in['x_train'][_class]:
                Xtrain[sample_index,:] = (dict_in['x_feature'][_class][training_index])
                ytrain[sample_index] = dict_in['y_label'][_class]
                sample_index+=1
        sample_index = 0         
        for _class_index,_class in enumerate(classes):
            for testing_index in dict_in['x_test'][_class]:
                Xtest[sample_index,:] = (dict_in['x_feature'][_class][testing_index])
                ytest[sample_index] = dict_in['y_label'][_class]
                sample_index+=1
        dict_in['y_truth']=ytest

        #train the model
        self.clf.fit(Xtrain,ytrain)

        #perform classification
        dict_in['y_pred']=self.clf.predict(Xtest)
        self.results.update(dict_in)
        
    class Factory:
        def create(self,ps_params,str_section):
            return Classify(ps_params,str_section)

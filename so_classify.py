#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj, sqrt,median, abs as nabs, exp, maximum as nmax
from numpy.fft import fftn, ifftn
from numpy.linalg import norm

from sklearn import svm
from sklearn.decomposition import PCA

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
        Class constructor for Classify
        """
        super(Classify,self).__init__(ps_params,str_section)
        self.S = OperatorComp(ps_params,
                              self.get_val('transforms',False))
        if len(self.S.ls_ops)==1: #avoid slow 'eval' in OperatorComp
            self.S = self.S.ls_ops[0] 
        self.classifier_method = self.get_val('method',False)
        self.feature_reduction = self.get_val('featurereduction',False)
        self.feature_sec_in = self.get_val('featuresectioninput',False)
        self.feature_sec_out = self.get_val('featuresectionoutput',False)
        if self.classifier_method[-3:]=='svc':
            kwprefix = 'kwsvc_'
            if self.classifier_method=='svc':
                self.clf = svm.SVC
            elif self.classifier_method=='linearsvc':
                self.clf = svm.LinearSVC
            else:
                raise ValueError('unknown svm method ' + self.classifier_method)    
            #get the keyword arguments dictionary to pass to the classifier constructor
            kwargs=self.get_keyword_arguments(kwprefix)
        elif self.classifier_method[-3:]=='pca':
            kwprefix = 'kwpca_'
            if self.classifier_method=='affinepca':
                self.clf = PCA
            else:
                raise ValueError('unknown pca method ' + self.classifier_method)    
        else:
            raise ValueError('unknown classification method ' + self.classifier_method)
        #manually fix some kwargs which shouldn't be lowercase...
        fixkeys=['C']
        for fixkey in fixkeys:
            if kwargs.has_key(fixkey.lower()):
                kwargs[fixkey] = kwargs[fixkey.lower()]
                kwargs.pop(fixkey.lower(),None)
        #instantiate the classifier    
        if self.isCallableWithArgs(self.clf.__init__,kwargs):
            self.clf=self.clf(**kwargs)
        else:
            raise ValueError('bad arguments for classifier ' + str(kwargs))    
        
    def solve(self,dict_in):
        """
        Takes an input object (ground truth partitioned classification set),
        computes features, and classifies.
        """
        super(Classify,self).solve()
        S = self.S
        feature_reduce = Scat().reduce #function handle 
        classes = dict_in['x'].keys()
        if self.feature_sec_in != '': #load the feature vectors from disk
            sec_input = sf.create_section(self.get_params(),self.feature_sec_in)
            dict_in['x_feature'] = sec_input.read(dict_in,return_val=True)
        else:   #no feature vector file specified, so compute
            met_output_obj = sf.create_section(self.get_params(),self.feature_sec_out)
            #generate the features if not previously stored from a prior run,takes a while...
            #TODO:parallelize
            for _class_index,_class in enumerate(classes):
                print 'generating scattering features for class ' + _class
                n_samples = len(dict_in['x'][_class])
                dict_in['x_feature'][_class]=(
                    [feature_reduce((S*dict_in['x'][_class][sample]).flatten(),
                                    method=self.feature_reduction) for 
                                    sample in xrange(n_samples)])
            #update and save
            met_output_obj.update(dict_in)
            self.results.save_metric(met_output_obj)

        #assumes each feature vector is the same size    
        dict_in['feature_vector_size'] = dict_in['x_feature'][classes[0]][-1].size
        Xtrain = np.zeros([dict_in['n_training_samples'],dict_in['feature_vector_size']],dtype='double') 
        Xtest = np.zeros([dict_in['n_testing_samples'],dict_in['feature_vector_size']],dtype='double') 
        
        #vector to hold the training labels
        ytrain = np.zeros(dict_in['n_training_samples'],dtype='int16')
        ytest = np.zeros(dict_in['n_testing_samples'],dtype='int16')
        sample_index = 0

        #generate the training and test data matrix (X) and label vectors (y)
        for _class_index,_class in enumerate(classes):
            for training_index in dict_in['x_train'][_class]:
                Xtrain[sample_index,:] = (dict_in['x_feature'][_class][training_index])
                ytrain[sample_index] = dict_in['y_label'][_class]
                sample_index+=1
        sample_index = 0
        ls_testing_index = []
        for _class_index,_class in enumerate(classes):
            for testing_index in dict_in['x_test'][_class]:
                Xtest[sample_index,:] = (dict_in['x_feature'][_class][testing_index])
                ytest[sample_index] = dict_in['y_label'][_class]
                ls_testing_index.append(testing_index)
                sample_index+=1
                
        dict_in['y_truth']=ytest
        dict_in['y_truth_sample_index']=np.array(ls_testing_index,dtype='int16')

        #rescaling X and y, since SVM is not scale invariant
        # Xtrain /= np.max(Xtrain)
        # Xtest /= np.max(Xtest)
        #train the model
        dict_in['pca_train']={}
        if self.classifier_method=='affinepca':
            #we must fit a model separately for each of the class subspaces
            for _class_index,_class in enumerate(classes):
                #xclass_features is n_samples (C) X n_features (\barSx_c)
                xclass_features=Xtrain[ytrain==dict_in['y_label'][_class],:]
                xclass_features_mean=np.mean(xclass_features,axis=0)
                #subtract the mean from each sample scattering vector (rows in xclass_features)
                #and do pca
                self.clf.fit(xclass_features)
                #store the components as a list in the pca_train dict
                #the first element being the mean, 
                #the second: the n_features x n_components array of principal components
                dict_in['pca_train'][_class]=[xclass_features_mean,self.clf._components]
        else:    
            self.clf.fit(Xtrain,ytrain)
            #perform classification/prediction on the test set
            dict_in['y_pred']=self.clf.predict(Xtest)
        self.results.update(dict_in)
        
    class Factory:
        def create(self,ps_params,str_section):
            return Classify(ps_params,str_section)

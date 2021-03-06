#!/usr/bin/python -tt
import numpy as np
from numpy import arange, conj, sqrt,median, abs as nabs, exp, maximum as nmax
from numpy.fft import fftn, ifftn
from numpy.linalg import norm
from sklearn import svm,linear_model
from sklearn.decomposition import PCA
from sklearn import preprocessing
import scipy.stats as ss

from py_utils.signal_utilities.scat import Scat
import py_utils.signal_utilities.sig_utils as su
from py_solvers.solver import Solver
from py_operators.operator_comp import OperatorComp
from py_utils.section_factory import SectionFactory as sf


class Classify(Solver):
    """General classify class which performs the feature computation, training,
    and validation.
    """
    def __init__(self,ps_params,str_section):
        """Class constructor for :class:`py_solvers.so_classify.Classify`.
        """
        super(Classify,self).__init__(ps_params,str_section)
        self.S = OperatorComp(ps_params,self.get_val('transforms',False))
        if len(self.S.ls_ops)==1: #avoid slow 'eval' in OperatorComp
            self.S = self.S.ls_ops[0]
        self.classifier_method = self.get_val('method',False)
        self.feature_reduction = self.get_val('featurereduction',False)
        self.feature_sec_in = self.get_val('featuresectioninput',False)
        self.class_sec_in = self.get_val('classsectioninput',False)
        self.feature_sec_out = self.get_val('featuresectionoutput',False)
        self.output = self.get_val('output',False)
        if self.classifier_method[-3:]=='svc':
            kwprefix = 'kwsvc_'
            if self.classifier_method=='svc':
                self.clf = svm.SVC
            elif self.classifier_method=='linearsvc':
                self.clf = svm.LinearSVC
            elif self.classifier_method=='linlogregsvc':
                self.clf = linear_model.LogisticRegression
            else:
                raise ValueError('unknown svm method ' + self.classifier_method)    
        elif self.classifier_method[-3:]=='pca':
            kwprefix = 'kwpca_'
            if self.classifier_method=='affinepca':
                self.clf = PCA
            else:
                raise ValueError('unknown pca method ' + self.classifier_method)    
        else:
            raise ValueError('unknown classification method ' + self.classifier_method)
        #get the keyword arguments dictionary to pass to the classifier constructor
        kwargs=self.get_keyword_arguments(kwprefix)
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
        super(Classify,self).solve()
        S = self.S
        feature_reduce = Scat().reduce #function handle to static method
        classes = sorted(dict_in['x'].keys())
        dict_in['class_labels']=classes
        if self.feature_sec_in != '': #load the feature vectors from disk
            sec_input = sf.create_section(self.get_params(), self.feature_sec_in)
            dict_in['x_feature'] = sec_input.read(dict_in, return_val = True)
            
            if self.class_sec_in != '': 
                #The class specification (csv), should already have 
                #been read in sec_input.
                #Now organize dict_in['x_feature'] into classes 
                #using dict_in['x'] as a reference
                #dict_in['x_feature'][exemplarid]->
                #dict_in['x_feature'][exemplarclass] 
                #where dict_in['x_feature']['exemplarclass'] is a list
                print 'organizing classes for this experiment...'
                for _class in classes:
                    dict_in['x_feature'][_class]=[]
                    for _exemplar in dict_in['x'][_class]:
                        exemplar_feature=dict_in['x_feature'].pop(_exemplar[0])
                        dict_in['x_feature'][_class].append(exemplar_feature)
        else:   #no feature vector file specified, so compute
            met_output_obj = sf.create_section(self.get_params(),
                                               self.feature_sec_out)
            #generate the features if not previously stored from a prior run
            #takes a while...
            for _class_index,_class in enumerate(classes):
                print 'generating scattering features for class ' + _class
                n_samples = len(dict_in['x'][_class])
                dict_in['x_feature'][_class]= (
                  [feature_reduce((S*dict_in['x'][_class][sample][1]).flatten(),
                                  method = self.feature_reduction)
                                  for sample in xrange(n_samples)])

        #assumes each feature vector is the same size for all exemplars...
        dict_in['feature_vec_sz'] = dict_in['x_feature'][classes[0]][-1].size
        Xtrain = np.zeros([dict_in['n_training_samples'],
                           dict_in['feature_vec_sz']],dtype='double')
        Xtest = np.zeros([dict_in['n_testing_samples'],
                          dict_in['feature_vec_sz']],dtype='double')
        
        #vector to hold the training labels
        ytrain = np.zeros(dict_in['n_training_samples'],dtype='int16')
        ytest  = np.zeros(dict_in['n_testing_samples'],dtype='int16')
        sample_index = 0
        print 'generating training/test data using pre-computed partitions...'
        #generate the training and test data matrix (X) and label vectors (y)
        for _class_index,_class in enumerate(classes):
            for train_ix in dict_in['x_train'][_class]:
                Xtrain[sample_index,:] = (dict_in['x_feature'][_class][train_ix])
                ytrain[sample_index] = dict_in['y_label'][_class]
                sample_index+=1
        sample_index = 0
        ls_testing_index = []
        ls_exemplar_id = []
        for _class_index,_class in enumerate(classes):
            for test_ix in dict_in['x_test'][_class]:
                Xtest[sample_index,:] = (dict_in['x_feature'][_class][test_ix])
                ytest[sample_index] = dict_in['y_label'][_class]
                ls_testing_index.append(test_ix)
                ls_exemplar_id.append(dict_in['x'][_class][test_ix][0])
                sample_index+=1
                
        dict_in['y_truth']=ytest
        dict_in['y_truth_sample_index']=np.array(ls_testing_index,dtype='int16')
        dict_in['exemplar_id']=ls_exemplar_id

        #rescaling X and y, since SVM is not scale invariant
        # Xtrain /= np.max(Xtrain)
        # Xtest /= np.max(Xtest)
        scaler = preprocessing.StandardScaler().fit(Xtrain)
        Xtrain=scaler.fit_transform(Xtrain)
        Xtest=scaler.transform(Xtest)
        dict_in['x_scaler_params']=scaler
        #train the model
        print 'fitting the model...'
        if self.classifier_method=='affinepca':
            dict_in['pca_train']={}
            #we must fit a model separately for each of the class subspaces
            for _cls_index,_cls in enumerate(classes):
                #xcls_feat is n_samples (C) X n_features (\barSx_c)
                xcls_feat=Xtrain[ytrain==dict_in['y_label'][_cls],:]
                xcls_feat_mean=np.mean(xcls_feat,axis=0)
                #subtract the mean from each sample scattering vector 
                #(rows in xcls_feat) and do pca
                self.clf.fit(xcls_feat-xcls_feat_mean)
                #store the components as a list in the pca_train dict
                #the first element being the mean (scattering class centroid), 
                #the second: the n_features x n_components array of PCs
                dict_in['pca_train'][_cls]=[xcls_feat_mean,self.clf.components_]
            #now fit the models by minimizing the error in the orthogonal linear 
            #space projection of classes
            #error matrix is (D+1) X ()
            #where D is the number of components in the pca decomposition
            pcaD=self.clf.components_.shape[0]+1
            err_mtx=np.zeros([pcaD,dict_in['n_testing_samples'],len(classes)])
            for _cls_index,_cls in enumerate(classes):      
                sx_minus_esxc=Xtest-dict_in['pca_train'][_cls][0]
                vc=dict_in['pca_train'][_cls][1]
                #calculate the orthoganal projection at each d in D
                #for the first row, sum energy along feature axis
                err_temp_1 = np.sum(np.abs(sx_minus_esxc)**2,axis=1)
                err_temp_rest = -(np.abs(vc.dot(sx_minus_esxc.T))**2)
                # pdb.set_trace()
                err_temp = np.cumsum(np.vstack([err_temp_1,err_temp_rest]),
                                     axis=0)**(.5)
                err_mtx[:,:,_cls_index]=err_temp
            #error matrix is now num_samplesXnum_classes
            err_mtx=err_mtx[-1,:,:]    
            #minimum D+1 error along class dimension
            error_mins=np.amin(err_mtx,axis=1)
            #now find the indices corresponding to these mins
            print 'making prediction...'
            dict_in['y_pred']=np.array([int(
                np.where(error_mins[j]==err_mtx[j,:])[0])
                for j in xrange(error_mins.shape[0])])
            dict_in['x_model_params']=dict_in['pca_train']
        else:#svm or some other method
            self.clf.fit(Xtrain,ytrain)
            #perform classification/prediction on the test set
            print 'making prediction...'
            if self.output=='probabilities':
                dict_in['y_pred']=self.clf.predict_proba(Xtest)
                for index,_class in enumerate(classes):
                    dict_in[_class]=dict_in['y_pred'][:,index]
            else:#non-probabilistic              
                dict_in['y_pred']=self.clf.predict(Xtest)
            dict_in['x_model_params']=self.clf
        self.results.update(dict_in)

    class Factory:
        def create(self,ps_params,str_section):
            return Classify(ps_params,str_section)

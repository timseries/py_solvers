# !/usr/bin/python -tt
import sys
import csv
import numpy as np
from numpy.linalg import norm
from collections import OrderedDict
from py_utils.parameter_struct import ParameterStruct
from py_utils.section_factory import SectionFactory as sf

from sklearn import preprocessing

import cPickle
import pdb 

def main():
    strmainpath='/home/tim/GoogleDrive/timothy.daniel.roberts@gmail.com/PhD/Projects/Classification/Data/GalaxyClassification' #workstation
    strmodelpath=strmainpath+'/images_training_rev1_formatted/training_model_small' #workstation
    ps_path='/home/tim/repos/py_solvers/applications/classification/galaxy_classification.ini'
    ps_params = ParameterStruct(ps_path)
    exp_list=['exp_1','exp_2','exp_3','exp_4','exp_5','exp_6','exp_7','exp_8','exp_9','exp_10','exp_11']
    class_dict={}
    class_dict[exp_list[0]]=['Class1.1','Class1.2','Class1.3']
    class_dict[exp_list[1]]=['Class2.1','Class2.2']
    class_dict[exp_list[2]]=['Class3.1','Class3.2']
    class_dict[exp_list[3]]=['Class4.1','Class4.2']
    class_dict[exp_list[4]]=['Class5.1','Class5.2','Class5.3','Class5.4']
    class_dict[exp_list[5]]=['Class6.1','Class6.2']
    class_dict[exp_list[6]]=['Class7.1','Class7.2','Class7.3']
    class_dict[exp_list[7]]=['Class8.1','Class8.2','Class8.3','Class8.4','Class8.5','Class8.6','Class8.7']
    class_dict[exp_list[8]]=['Class9.1','Class9.2','Class9.3']
    class_dict[exp_list[9]]=['Class10.1','Class10.2','Class10.3']
    class_dict[exp_list[10]]=['Class11.1','Class11.2','Class11.3','Class11.4','Class11.5','Class11.6']
    class_dict = OrderedDict(sorted(class_dict.items(), key=lambda t: t[0]))
    #do classification on testing set with unknown labels
    sec_input = sf.create_section(ps_params,'Input2')
    dict_in={}
    dict_in['x_feature']=sec_input.read(dict_in,return_val=True)
    for exp in exp_list:
        with open(strmainpath + '/' + exp + '_predict.csv', 'wb') as csvfile:
            galaxywriter = csv.writer(csvfile)
            Xtest=np.zeros([len(dict_in['x_feature'].keys()),241])
            for idx,galaxy_id in enumerate(dict_in['x_feature'].keys()):
                #aggregate the test matrix
                Xtest[idx,:]=dict_in['x_feature'][galaxy_id]
            #load the model for this experiment
            scaler = preprocessing.StandardScaler().fit(Xtest)
            Xtest=scaler.fit_transform(Xtest)
            filehandler = open(strmodelpath + '/' + exp + '_x_model_params.pkl', 'r') 
            model = cPickle.load(filehandler)
            filehandler.close()
            #make prediction
            ypreds=model.predict_proba(Xtest)
            #write to the currently open csv file
            galaxywriter.writerow(['GalaxyID']+class_dict[exp]) #header
            for idx,galaxy_id in enumerate(dict_in['x_feature'].keys()):
                galaxywriter.writerow([galaxy_id]+list(ypreds[idx,:]))

    #load in the features
if __name__ == "__main__":
    main()
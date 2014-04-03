#! /home/zelda/tr331/ENV/bin/python
#$ -S /home/zelda/tr331/ENV/bin/python

#ad;lfkja;lfke   !/usr/bin/python -tt
import csv
import matplotlib.image as mpimg
import png
from scipy import misc
import numpy as np
import cPickle

from py_utils.signal_utilities.scat import Scat
from py_utils.parameter_struct import ParameterStruct
from py_utils.signal_utilities.sig_utils import rgb2gray
from py_utils.section_factory import SectionFactory as sf 

import pdb

def main():
    # strpath='/home/tim/GoogleDrive/timothy.daniel.roberts@gmail.com/PhD/Projects/Classification/Data/GalaxyClassification/' #workstation
    strpath='/home/zelda/tr331/Projects/GalaxyChallenge' #yoshi
    training_dir='images_training_rev1/'
    save_dir='images_training_rev1_formatted/'
    exp_list=['exp1','exp2','exp3','exp4','exp5','exp6','exp7','exp8','exp9','exp10','exp11']
    for index,exp in enumerate(exp_list):
        exp_list[index]='class_csv_'+exp
    class_dict={}
    class_cols={}
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
    class_dict[exp_list[10]]=['Class11.1','Class11.2','Class11.3']
    tgt_size=[128,128]
    feature_reduce = Scat().reduce #function handle 
    gen_csv=0
    gen_bw_cropped_images=1
    gen_feature_vector_files=1
    if gen_feature_vector_files:
        feature_vector={}#dict, keys are galaxyids'
        ps_path=strpath+save_dir+'galaxy_params.ini'
        ps_params = ParameterStruct(ps_path)
        S = sf.create_section(ps_params,'Transform2')
    with open(strpath+'training_solutions_rev1.csv', 'rb') as csvfile:
        galaxyreader = csv.reader(csvfile)
        count = 0 #to skip the first row
        for row in galaxyreader:
            if count == 0:#skip
                header=row
                #get the column numbers (as lists) corresponding to each experiment
                for exp in exp_list:
                    class_cols[exp]=[header.index(_class) for _class in class_dict[exp]]
            else:    
                print 'processing galaxy ' + row[0]
                #decode the row data and separate into classes
                if gen_csv:
                    for exp in exp_list:
                        with open(strpath+exp+'.csv', 'ab') as csvfile2:
                            expwriter = csv.writer(csvfile2)
                            #pull out the probabilities corresponding to each class
                            class_probs=np.array([row[column] for column in class_cols[exp]])
                            #find the maximal class 
                            _class=class_dict[exp][np.argmax(class_probs)]
                            #output the maximal class
                            expwriter.writerow([row[0]] + [_class])

                galaxyid=row[0]
                #now open the file, and do some adjustments
                traingalaxyfile=strpath+training_dir+galaxyid+'.jpg'
                savegalaxyfile=strpath+save_dir+galaxyid+'.png'
                #open the galaxies and resave them if specified
                if gen_bw_cropped_images:      
                    galaxydata=mpimg.imread(traingalaxyfile) 
                    newgalaxy=np.zeros(tgt_size)
                    #get the cropping region
                    st0=(galaxydata.shape[0]-tgt_size[0])/2
                    st1=(galaxydata.shape[1]-tgt_size[1])/2
                    SL0=slice(st0,st0+tgt_size[0],None)
                    SL1=slice(st1,st1+tgt_size[1],None)
                    #convert color-> rgb
                    galaxydata=rgb2gray(galaxydata)
                    newgalaxy=galaxydata[SL0,SL1]
                    f = open(savegalaxyfile,'wb')
                    w = png.Writer(*(newgalaxy.shape[1],newgalaxy.shape[0]),greyscale=True)
                    newgalaxy=newgalaxy/np.max(newgalaxy)*255
                    w.write(f,newgalaxy)
                    f.close()
                if gen_feature_vector_files:
                    x_data = misc.imread(savegalaxyfile)
                    feature_vector[galaxyid]=(feature_reduce((S*x_data).flatten(),method='average'))
            count+=1
                                                      

    #now generate the features vector files for each experiment
    if gen_feature_vector_files:        
        feature_file=strpath+save_dir+'allfeatures.pkl'
        filehandler = open(feature_file, 'wb')
        cPickle.dump(feature_vector, filehandler) 
        filehandler.close()
        #reopen
        filehandler = open(feature_file, 'r') 
        feature_vector = cPickle.load(filehandler)
        filehandler.close()
        for exp in [exp_list[0]]:
            print 'creating feature vector file for ' + exp
            feature_vector_exp={}
            for _class in class_dict[exp]:#initialize with empty feature list
                feature_vector_exp[_class]=[]
            feature_file_exp=exp+'_features.pkl'
            exp_csv_file=exp + '_small.csv'
            with open(strpath+save_dir+exp_csv_file, 'rb') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    feature_vector_exp[row[1]].append(feature_vector[row[0]])
            filehandlerexp = open(strpath+save_dir+feature_file_exp, 'wb') 
            cPickle.dump(feature_vector_exp, filehandlerexp) 
            filehandlerexp.close()
            
if __name__ == "__main__":
    main()
# !/usr/bin/python -tt
import sys
import csv
from collections import OrderedDict
import numpy as np
from numpy.linalg import norm

import pdb 

def main():
    exp_list=['exp_1','exp_2','exp_3','exp_4','exp_5','exp_7','exp_9','exp_6','exp_8','exp_10','exp_11']
    output_keys = ['GalaxyID']
    class_dict={}
    class_dict['exp_1']=['Class1.1','Class1.2','Class1.3']
    class_dict['exp_2']=['Class2.1','Class2.2']
    class_dict['exp_3']=['Class3.1','Class3.2']
    class_dict['exp_4']=['Class4.1','Class4.2']
    class_dict['exp_5']=['Class5.1','Class5.2','Class5.3','Class5.4']
    class_dict['exp_6']=['Class6.1','Class6.2']
    class_dict['exp_7']=['Class7.1','Class7.2','Class7.3']
    class_dict['exp_8']=['Class8.1','Class8.2','Class8.3','Class8.4','Class8.5','Class8.6','Class8.7']
    class_dict['exp_9']=['Class9.1','Class9.2','Class9.3']
    class_dict['exp_10']=['Class10.1','Class10.2','Class10.3']
    class_dict['exp_11']=['Class11.1','Class11.2','Class11.3','Class11.4','Class11.5','Class11.6']
    class_dict = OrderedDict(sorted(class_dict.items(), key=lambda t: int(t[0][4:])))
    for exp in class_dict.keys():
        output_keys += class_dict[exp]
    strpath='/home/tim/GoogleDrive/timothy.daniel.roberts@gmail.com/PhD/Projects/Classification/Data/GalaxyClassification' #workstation
    dict_result={} #each entry of this is a dictionary
    #fill-in the galaxyids
    count=0
    with open(strpath + '/' + exp_list[0] + '_predict.csv', 'rb') as csvfile:
        galaxyreader = csv.reader(csvfile)
        for row in galaxyreader: 
            if count==0:
                headers=row
                exemplar_id_col=headers.index('GalaxyID')
            else:
                dict_result[row[exemplar_id_col]]={}
                dict_result[row[exemplar_id_col]]['GalaxyID']=row[exemplar_id_col]
            count+=1    
    #fill in the columns, using the tree rules to do computations        
    for exp in exp_list: #do multiple passes in the file, down the tree
        print exp
        count=0
        with open(strpath + '/' + exp + '_predict.csv', 'rb') as csvfile:
            galaxyreader = csv.reader(csvfile)
            for row in galaxyreader: 
                if count==0:
                    headers=row
                    exemplar_id_col=headers.index('GalaxyID')
                    metrics=headers[exemplar_id_col:]
                else:
                    # pdb.set_trace()
                    current_galaxy_dict = dict_result[row[exemplar_id_col]]
                    for _class in class_dict[exp]:
                        _class_index = headers.index(_class)
                        value = np.float(row[_class_index])
                        if exp=='exp_1':
                            value = value * 1.0
                        elif exp=='exp_2':
                            value = value * current_galaxy_dict['Class1.2']
                        elif exp=='exp_3':
                            value = value * current_galaxy_dict['Class2.1']
                        elif exp=='exp_4':
                            value = value * (current_galaxy_dict['Class3.1']+current_galaxy_dict['Class3.2'])
                        elif exp=='exp_5':
                            value = value * current_galaxy_dict['Class4.2']
                        elif exp=='exp_6':
                            value = value * (current_galaxy_dict['Class5.1']+
                                             current_galaxy_dict['Class5.2']+
                                             current_galaxy_dict['Class5.3']+
                                             current_galaxy_dict['Class5.4']+
                                             current_galaxy_dict['Class7.1']+
                                             current_galaxy_dict['Class7.2']+
                                             current_galaxy_dict['Class7.3'])
                        elif exp=='exp_7':
                            value = value * current_galaxy_dict['Class1.1']
                        elif exp=='exp_8':
                            value = value * current_galaxy_dict['Class6.1']
                        elif exp=='exp_9':
                            value = value * current_galaxy_dict['Class2.1']
                        elif exp=='exp_10':
                            value = value * current_galaxy_dict['Class4.1']
                        elif exp=='exp_11':
                            value = value * (current_galaxy_dict['Class10.1']+
                                             current_galaxy_dict['Class10.2']+
                                             current_galaxy_dict['Class10.3'])
                        else:
                            print 'invalid experiment ' + exp    
                            
                        dict_result[row[exemplar_id_col]][_class]=value
                    #need to renormalize class 6
                    
                    if exp=='exp_6':
                        normalizer=0
                        for _class in class_dict[exp]:
                            normalizer+=dict_result[row[exemplar_id_col]][_class]
                        for _class in class_dict[exp]:    
                            dict_result[row[exemplar_id_col]][_class]/=normalizer
                count+=1    
    with open(strpath + '/' + 'final_output.csv', 'wb') as csvfile:
        galaxywriter = csv.DictWriter(csvfile,output_keys)
        for galaxy_id in dict_result.keys():
            galaxywriter.writerow(dict_result[galaxy_id])
    print output_keys                
    #now compute the rmse by scanning through the solutions
if __name__ == "__main__":
    main()

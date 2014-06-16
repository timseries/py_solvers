# !/usr/bin/python -tt
import sys
import csv
import numpy as np
from numpy.linalg import norm

import pdb 

def main():
    strpath='/home/tim/GoogleDrive/timothy.daniel.roberts@gmail.com/PhD/Projects/Classification/Data/GalaxyClassification/images_training_rev1_formatted/training_model_small' #workstation
    gtcsv='/home/tim/GoogleDrive/timothy.daniel.roberts@gmail.com/PhD/Projects/Classification/Data/GalaxyClassification/training_solutions_rev1.csv' #workstation
    count=0
    dict_result={}
    with open(strpath + '/' + sys.argv[1], 'rb') as csvfile:
        galaxyreader = csv.reader(csvfile)
        #build a dictionary of samples
        for row in galaxyreader:
            if count==0:
                headers=row
                exemplar_id_col=headers.index('exemplar_id')
                metric_gen=xrange(exemplar_id_col+1,exemplar_id_col+3)
            else:
                dict_result[row[exemplar_id_col]]=np.asarray([np.float(row[j]) for j in metric_gen])
            count+=1        
    #now compute the rmse by scanning through the solutions
        count=0
        diff_list=[]
        with open(gtcsv, 'rb') as csvfile:
            galaxyreader = csv.reader(csvfile)
            for row in galaxyreader:
                if count==0:
                    headers2=row
                    exemplar_id_col=headers2.index('GalaxyID')
                    metric_gen=xrange(exemplar_id_col+1,exemplar_id_col+3)#only comparing the first 3 class probs
                else:
                    exemplar_id=row[exemplar_id_col]
                    gt=np.asarray([np.float(row[j]) for j in metric_gen])
                    if dict_result.has_key(exemplar_id):
                        diff_list.append(dict_result[exemplar_id]-gt)
                count+=1
        sse=np.hstack(diff_list)            
        mse=1.0/sse.size*np.sum(sse**2)    
        rmse=np.sqrt(mse)
        print 'rmse is ' + str(rmse)
if __name__ == "__main__":
    main()

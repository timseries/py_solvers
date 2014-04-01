#!/usr/bin/python -tt
import csv
import matplotlib.image as mpimg
import png
import numpy as np

def main():
    strpath='/home/tim/GoogleDrive/timothy.daniel.roberts@gmail.com/PhD/Projects/Classification/Data/CellPatternChallenge/'
    trainpath=strpath+'ICIP2013_training_1.0/train/'
    class_list=['homogeneous','speckled','nucleolar','centromere','golgi','numem']
    savepath=strpath+'Organized/'
    tgt_size=[128,128]
    with open(strpath+'ICIP2013_training_1.0/gt_training.csv', 'rb') as csvfile:
        cellreader = csv.reader(csvfile)
        count = 0 #to skip the first row
        for row in cellreader:
            if count == 0:#skip
                pass
            else:    
                cellid=int(row[0])
                prepadzeros = '0'*int(5-np.log10(cellid+.001))
                cellclass=row[1].lower()
                prepad_classprefix='T0'+str(class_list.index(cellclass)+1)+'_'
                celldir=prepad_classprefix+cellclass
                traincellfile=trainpath+prepadzeros+str(cellid)
                traincellmaskfile=traincellfile+'_Mask'+'.png'
                traincellfile+='.png'
                savecellpath=savepath+celldir+'/'
                savecellfile=savecellpath+str(cellid) + '.png'
                #open the cell
                celldata=mpimg.imread(traincellfile) 
                cellmaskdata=mpimg.imread(traincellmaskfile) 
                newcell=np.zeros(tgt_size)
                #paste in the masked data centered roughly in the new image
                st0=(tgt_size[0]-celldata.shape[0])/2
                st1=(tgt_size[1]-celldata.shape[1])/2
                SL0=slice(st0,st0+celldata.shape[0],None)
                SL1=slice(st1,st1+celldata.shape[1],None)
                newcell[SL0,SL1]=celldata*cellmaskdata
                f = open(savecellfile,'wb')
                w = png.Writer(*(newcell.shape[1],newcell.shape[0]),greyscale=True)
                w.write(f,newcell/np.max(newcell)*255)
                f.close()
            count+=1
            
if __name__ == "__main__":
    main()
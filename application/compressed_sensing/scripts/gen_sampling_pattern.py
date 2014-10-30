#based on M. Lustigs SparseMRI genSampling and genPDF routines
import numpy as np
from numpy.fft import fftn, ifftn
from numpy import abs as nabs, max as nmax, floor,zeros
import matplotlib.pyplot as plt

def main():   
    mask_path= '/home/tim/repos/py_solvers/application/data/velocity_imaging/velimg1_sample_mask.npz'
    pdf=genPDF([1,256],p=2.5,pctg=0.3,distType=2,radius=0.1) 
    nitn = 100
    tol = 1
    samplemask, stat, actpctg = genSampling(pdf,nitn,tol)
    #extedn this to a 2d mask
    new_mask = np.tile(samplemask,(256,1))
    np.savez_compressed(mask_path,new_mask)
    
    
def genPDF(imSize,p,pctg,distType=2,radius=0,seed=0):
    minval=0
    maxval=1
    val = 0.5

    if len(imSize)==1:
        imSize = [imSize,1]
    sx = imSize[0]
    sy = imSize[1]
    PCTG = np.floor(pctg*sx*sy)
    if not np.any(np.asarray(imSize)==1):
        x,y= np.meshgrid(np.linspace(-1,1,sy),np.linspace(-1,1,sx))
        if distType == 1:
            r = np.fmax(nabs(x),nabs(y))
        else:
            r = sqrt(x**2+y**2)
            r = r/nmax(nabs(r.flatten()))		
    else:
        r = nabs(np.linspace(-1,1,max(sx,sy)))

    idx = np.where(r<radius)
    pdf = (1-r)**p
    pdf[idx] = 1
    if np.floor(sum(pdf.flatten())) > PCTG:
      raise ValueError('infeasible without undersampling dc, increase p')

    # begin bisection
    while 1:
        val = minval/2.0 + maxval/2.0
        pdf = (1-r)**p + val
        pdf[pdf>1] = 1
        pdf[idx]=1
        N = np.floor(sum(pdf.flatten()))
        if N > PCTG:
            maxval=val
        if N < PCTG:
            minval=val
        if N==PCTG:
            break
    return pdf


def genSampling(pdf,nitn,tol):
    pdf[pdf>1] = 1
    K = np.sum(pdf.flatten())

    minIntr = 1e99
    minIntrVec = zeros(pdf.shape)
    stat = np.zeros(nitn,)
    for n in np.arange(0,nitn):
        tmp = zeros(pdf.shape)
        while abs(np.sum(tmp.flatten()) - K) > tol:
            tmp = rand(*pdf.shape)<pdf
            
        TMP = ifft2(tmp/pdf)
        if nmax(nabs(TMP.flatten()[1:])) < minIntr:
            minIntr = nmax(nabs(TMP.flatten()[1:]))
            minIntrVec = tmp
        stat[n] = nmax(nabs(TMP.flatten()[1:]))

    actpctg = np.sum(minIntrVec.flatten())/float(minIntrVec.size)
    mask = minIntrVec
    return mask, stat, actpctg
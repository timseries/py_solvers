#script to generate a spatialmask boundary 
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt2d
from mpldatacursor import datacursor

#input spatialmask path
#parameters vor velimg1
data_path ='/home/tim/repos/py_solvers/application/data/velocity_imaging/velimg1.npz'
data = np.load(data_path)['arr_0']
conjprod = 1
frame_order='reverse'
frame_order = [0, 1]
extrafftshift=1
for frame in xrange(2):
    data[:,:,frame]=fftn(fftshift(data[:,:,frame]))
if extrafftshift:
    for frame in xrange(2):
        data[:,:,frame]=fftshift(data[:,:,frame])
        
# plt.hist(np.abs(data.flatten()),bins=100)
if frame_order!='forward':
    frame_order=[1, 0]
#need to do some preprocessing of this datset first...
if conjprod:
    new_x = np.sqrt(data[:,:,frame_order[0]] * 
             conj(data[:,:,frame_order[1]]))

spatialmask = np.zeros(new_x.shape)
thresh= (31250+259315)/4.1
spatialmask[np.abs(new_x)>thresh] = 1
spatialmask_mask = np.ones(spatialmask.shape)
spatialmask_mask[200:220,75:85] = spatialmask[200:220,75:85]
spatialmask = medfilt2d(spatialmask,[3, 3])
spatialmask *= spatialmask_mask
spatialmask = medfilt2d(spatialmask,[3, 3])
spatialmask = np.asarray(spatialmask,dtype='bool')  
spatialmask[212,81]=True

fixmask = []
#tblr tuples
fixmask.append([223,256,198,205])
fixmask.append([0,16,198,205])
fixmask.append([29,30,200,201])
fixmask.append([218,219,200,201])
fixmask.append([153,154,11,12])
fixmask.append([8,10,163,165])
fixmask.append([187,188,226,227])
fixmask.append([216,217,200,201])
fixmask.append([21,22,189,190])
fixmask.append([17,18,182,183])
fixmask.append([217,218,199,200])
fixmask.append([231,232,174,175])
fixmask.append([222,223,62,63])
fixmask.append([194,195,69,70])
fixmask.append([64,65,39,40])
for fixmaskel in fixmask:
    spatialmask[fixmaskel[0]:fixmaskel[1],fixmaskel[2]:fixmaskel[3]]=False
    
#save boundary spatialmask
# np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/2dsine_spatial_spatialmask_boundary.npz',boundary_spatialmask*spatialmask)
spatialmask_path= '/home/tim/repos/py_solvers/application/data/velocity_imaging/velimg1_spatial_mask.npz'
np.savez_compressed(spatialmask_path,spatialmask)

spatialmask=np.load(spatialmask_path)['arr_0']
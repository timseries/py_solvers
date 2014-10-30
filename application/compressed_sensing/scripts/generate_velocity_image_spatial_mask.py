#script to generate a mask boundary 
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt2d

#input mask path
# mask_path = '/home/tim/repos/py_solvers/application/data/velocity_imaging/2dsine_spatial_mask.npz'
data_path ='/home/tim/repos/py_solvers/application/data/velocity_imaging/velimg1.npz'
data = np.load(data_path)['arr_0']
conjprod = 1
frame_order='forward'
frame_order = [0, 1]
extrafftshift=1
for frame in xrange(2):
    data[:,:,frame]=fftn(fftshift(data[:,:,frame]))
if extrafftshift:
    for frame in xrange(2):
        data[:,:,frame]=fftshift(data[:,:,frame])
        
plt.hist(np.abs(data.flatten()),bins=100)
if frame_order!='forward':
    frame_order=[1, 0]
#need to do some preprocessing of this datset first...
if conjprod:
    new_x = np.sqrt(data[:,:,frame_order[0]] * 
             conj(data[:,:,frame_order[1]]))

mask = np.zeros(new_x.shape)
thresh= (31250+259315)/2.0
mask[np.abs(new_x)>thresh] = 1
mask = medfilt2d(mask,[3, 3])
mask = np.asarray(mask,dtype='bool')
#save boundary mask
# np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/2dsine_spatial_mask_boundary.npz',boundary_mask*mask)
mask_path= '/home/tim/repos/py_solvers/application/data/velocity_imaging/velimg1_spatial_mask.npz'
np.savez_compressed(mask_path,mask)

mask=np.load(mask_path)['arr_0']
import numpy as np
import matplotlib.pyplot as plt
imsize=np.array([168,168]) 
phase = np.zeros(imsize)
mag = np.zeros(imsize)

vess1_rad=20
vess1_loc=np.array([40,40])
vess1_max_phase=6

vess2_rad=30
vess2_loc=np.array([80,80])
vess2_max_phase=3

cx,cy = mgrid[0:imsize[0],0:imsize[1]]

for ix_ in xrange(2):
    if ix_==0:
        rad = vess1_rad
        loc = vess1_loc
        max_phase = vess1_max_phase
    else:    
        rad = vess2_rad
        loc = vess2_loc
        max_phase = vess2_max_phase

    circle_dist = (cx-loc[0])**2 + (cy-loc[1])**2
    circle = (circle_dist < (rad)**2)
    vel = -(circle_dist * circle)
    vel += np.abs(np.min(vel))
    vel *= circle
    vel = np.asfarray(vel)
    vel /= np.max(vel)
    vel *= max_phase

    if ix_==0:
        vess1_mag = circle
        vess1_phase = vel
    else:
        vess2_mag = circle
        vess2_phase = vel
        
mag+=vess1_mag
mag+=vess2_mag

phase+=vess1_phase
phase+=vess2_phase

plt.figure()    
plt.imshow(phase)
plt.figure()    
plt.imshow(mag)

signal = mag * np.exp(1j * phase)
mask = mag

np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/vessels_fully_sampled_data.npz',signal)
np.savez_compressed('/home/tim/repos/py_solvers/application/data/velocity_imaging/vessels_spatial_mask.npz',mask)
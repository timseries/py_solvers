For datasets starting with 'P', ground truth is available.

phantom.ics and phantom.tif
Ground truth (cropped to 192x192x64) - use this for numerical comparisons to ground truth

phantom_padded.tif
Ground truth (padded to 320x320x64) - feed this into forward model

psf.tif 
Point spread function, same psf corresponding to the numbered trainingevaluation example for the challenge

benchmark_recon.tif
The result of running the reference RL deblur algorithm for 100 iterations

slice#.eps 
A slice take from the image stack at z=#.
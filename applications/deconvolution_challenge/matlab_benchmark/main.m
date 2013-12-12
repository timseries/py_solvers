%should be in the top level of these directories in MATLAB current directory (matlab_benchmark)
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Operators/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Reconstruction/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Metrics/'])
addpath(['/home/tim/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Misc/'])

str_dir = '../data/Training/';
str_measurements = 'Measurements/';
str_psfs = 'PSFs/';
str_case = '0';
ary_ground_truth = imreadstack([str_dir str_measurements str_case '.tif']);
ary_psf = imreadstack([str_dir str_psfs str_case '_psf.tif']);
%0
mp = 239.6;
b = 15.8;
stdev = 9.7;
seed = 1;
%observe
%[y,f,fb]=ForwardModel3D(ary_ground_truth,ary_psf,mp,b,stdev,seed);
%reverse quantize the measurements y
y = double(ary_ground_truth);
%[x0,fun_val,QS]=RLdeblur3D(y,ary_psf,'img',ary_ground_truth,'imgb',fb,'iter',100,'verbose',true,'showfig',true)
[x0,fun_val,QS]=RLdeblur3D(ary_ground_truth,ary_psf,'iter',30,'verbose',true,'showfig',true);
implay(uint8(x0));

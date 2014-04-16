addpath('~/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Metrics/FourierMetrics/')
addpath('~/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/FourierMetrics/')
addpath('~/repos/py_solvers/applications/deconvolution_challenge/matlab_benchmark/Misc/')
str_input_file='/home/tim/repos/scratch/deconv_challenge_output/p0/2014-04-16_22-08-02/p0_x_n_2.tif';
x_n_2=imreadstack(str_input_file);
str_input_file='/home/tim/repos/scratch/deconv_challenge_output/p0/2014-04-16_22-08-02/p0_x_1.tif';
x=imreadstack(str_input_file);

disp('nmisec')
nmisec(x_n_2,x)



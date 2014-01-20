function [y,f,fb]=ForwardModel3D(f,h,mp,b,stdev,seed)

% Adopted Observation model:
%
%  y=Q[Poiss(h*f+b)+w]. 
%
% where Q is a function that quantizes the output.
%
% ========================== INPUT PARAMETERS (required) ==================
% Parameter     Values
% name          and description
% =========================================================================
% f             Reference image.
% h             Blur kernel (PSF).
% mp            Maximum number of photons per voxel. max(k(f*h))=mp =>
%               k=mp/max(f*h). f'=k*f;
%======================== OPTIONAL INPUT PARAMETERS =======================
% b             Constant value of the background (Default:0).
% stdev         Standard deviation for the gaussian noise added to the 
%               Poisson measurements (Default:0).
% seed          Seed for the random noise generator.
%============================ OUTPUT PARAMETERS ===========================
% Parameters    Values
% y             Degraded Image according to the adopted forward model.
% f             Normalized Reference image and appropriately cropped to 
%               match the size of the measurements.
% fb            fb=(h*f)+b (h:psf, b:constant value for the background)
% ============k=============================================================

%Authors: stamatis.lefkimmiatis@epfl.ch, cedric.vonesch@epfl.ch (Biomedical Imaging Group)

if nargin < 6
  seed=1;
end

if nargin < 5
  stdev=0;
end

if nargin < 4
  b=0;
end

if mp < b
  error('ForwardModel3D: background value should be smaller than the mp parameter.');
end

if b < 0
  error('ForwardModel3D: background value should be non-negative');
end

%We perform a discrete convolution where we keep only the valid part of the result.
r=Direct(h,f);
k=mp/max(r(:));
r=k*r;
f=k*f;

stream = RandStream('mcg16807', 'Seed', seed);
RandStream.setGlobalStream(stream);

if stdev==0
  w=0;
else
  w=stdev*randn(size(r));
end

fb=r+b;
f=Crop(f,r);
disp(mean(f(:)));
y=int16(poissrnd(fb)+w);
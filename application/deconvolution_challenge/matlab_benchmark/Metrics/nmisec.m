function score=nmisec(r,f,y)

%Normalized mean integrated squared error

% ========================== INPUT PARAMETERS (required) ==================
% Parameters    Values and description
% =========================================================================
% r             processed image-stack.
% f             ground-truth image-stack.
% y             y=(h*f)+b (h:psf, b:constant value for the background)
% ========================== OUTPUT PARAMETERS ============================
% score         Modified version of the normalized mean integrated squared 
%               error, which is appropriate for the more general deblurring
%               problem.
% =========================================================================

%Author: stamatis.lefkimmiatis@epfl.ch (Biomedical Imaging Group)


%r:reconstructed image
%f:groundtruth
%y=Kf+b; The blurred version of f plus background.

%A version of nmise where it corrects the problem when f is zero at some
%pixels. 

if ~isequal(size(y), size(f)), error('non-matching y-f'); end
if ~isequal(size(r), size(f)), error('non-matching r-f'); end


score=((r-f).^2)./y;
score=mean(score(:));


function [x0,fun_val,QS]=RLdeblur3D(y,h,varargin)

%Image deblurring with Richardson-Lucy algorithm.

% ========================== INPUT PARAMETERS (required) ==================
% Parameters    Values  description
% =========================================================================
% y             Noisy blured image.
% h             Blur kernel (PSF).
% ======================== OPTIONAL INPUT PARAMETERS ======================
% Parameters    Values description
% =========================================================================
% img           Original Image. (For the compution of the Quality metrics)
% imgb          Blurred Image (For the compution of the Normalized mean 
%               integrated squared error). It is the third output argument
%               of the ForwardModel3D.m
% x_init        Initial guess for x. (Default: [])
% iter          Number of iterations (Default: 100)
% tol           Stopping threshold: Relative normed difference between two
%               successive iterations (Default:1e-5)
% verbose       If verbose is set on then info for each iteration is
%               printed on screen. (Default: false)
% showfig       If showfig is set to true, the maximum intensity projection 
%               of the result of the deconvolution in each iteration is 
%               shown on screen. (Default: false)
% b             Constant value for the background (Default: 0)
% K             constants in the SSIM index formula (see ssim_index.m).
%               default value: K = [0.01 0.03]
% window        local window for statistics (see ssim_index.m). default 
%               window is Gaussian given by window = fspecial('gaussian', 11, 1.5);
% L             dynamic range of the images. default: max(X2(:))-min(X2(:))
% ismetric      If ismetric is set to true then the quality metrics are
%               computed for every iteration. (Default: false) (Note: If
%               this option is set to true the script can be very slow)
% =========================================================================

%Author: stamatis.lefkimmiatis@epfl.ch (Biomedical Imaging Group)

[x_init,iter,verbose,showfig,tol,img,imgb,b,K,window,L,ismetric]=...
  process_options(varargin,'x_init',[],'iter',100,'verbose',true,...
  'showfig',false,'tol',1e-5,'img',[],'imgb',[],'b',0,'K',[0.01 0.03],...
  'window', fspecial('gaussian', 11, 1.5),'L',[],'ismetric',false);


if nargout > 2 && ismetric
  nbins=64;
  Z=zeros(iter,1);
  img_hat=fftn(img);
  metric=FourierMetricsConstructor(size(y),nbins);
  QS=struct('ISNR',Z,'RSNR',Z,'nmise_metric',Z,'ssim_metric',[Z Z],...
    'FSC_metric',repmat(Z,1,nbins),'ER_metric',repmat(Z,1,nbins));  
else
  QS=[];
end


if b < 0
  error('RLdeblur3D: Background must have a non-negative value');
end


% Operators
Bop = @(x) Direct(h, x); % Function handle that corresponds to the blurring operator
AdjBop = @(y) Adjoint(h, y); % Function handle that corresponds to the adjoint of the blurring operator

%x0 Initialization
if isempty(x_init)
    disp(['mean y to adjoint: ', num2str(mean(y(:)))]);
    x_init=Adjoint(h, y);
end

x=x_init;

x0_mat=x_init;
% Normalization constant
gamma = AdjBop(ones(size(y)));

fun_val=zeros(iter,1);

count=0;
if verbose
  fprintf('\t\t****************************************\n');
  fprintf('\t\t** Deconvolution with Richardson-Lucy **\n');
  fprintf('\t\t****************************************\n');
  fprintf('#iter       fun-val      relative-dif     ISNR\t  RSNR\t   NMISE\t  SSIM_mean\t   SSIM_min\n')
  fprintf('==========================================================================================\n');
end
for i=1:iter
  div=y./(Bop(x)+b);
  div(isnan(div))=0;
  xnew=AdjBop(div).*x;
  xnew = xnew./gamma;

  disp(['x_n min: ' num2str(min(xnew(:)))]);
  disp(['x_n max: ' num2str(max(xnew(:)))]);
  
  fun_val(i)=cost(y,Bop,xnew,b);
     
   re=norm(xnew(:)-x(:))/norm(x(:));%relative error
  if (re < tol)
    count=count+1;
  else
    count=0;
  end
  
  
  if verbose 
    if ~isempty(img) && ismetric
	  x0 = Crop(xnew, y);
      if isempty(L)
        L=max(img(:))-min(img(:));
      end

	  xreg = AffineRegression(x0, img);
      QS.ISNR(i)=20*log10(norm(y(:)-img(:))/norm(x0(:)-img(:)));
      QS.RSNR(i)=20*log10(norm(y(:)-img(:))/norm(xreg(:)-img(:)));
      QS.RSNR(i)=0;
      if isempty(imgb)
        QS.nmise_metric(i)=nan;
      else
        QS.nmise_metric(i)=nmisec(x0,img,imgb);
      end
      [ssim_mean,ssim_min]=ssim3D(x0,img,K,window,L);
      QS.ssim_metric(i,:)=[ssim_mean, ssim_min];
      x0_hat=fftn(x0);
      QS.FSC_metric(i,:)=FourierShellCorrelation(metric,x0_hat,img_hat);
      QS.ER_metric(i,:)=RelativeEnergyRegain(metric,x0_hat,img_hat);
      
      % printing the information of the current iteration
      fprintf('%3d \t %3.5f \t %3.5f \t %3.5f \t %3.5f \t%3.5f \t%3.5f \t%3.5f\n',i,...
      fun_val(i),re,QS.ISNR(i),QS.RSNR(i),QS.nmise_metric(i),QS.ssim_metric(i,1),QS.ssim_metric(i,2));
    else
      fprintf('%3d \t %3.5f \t %3.5f\n',i,fun_val(i),re);
      %fprintf('%3d \t %3.5f\t%1.5f\n',i,fun_val(i),re);
    end
  end
  x=xnew; 
    
  if showfig
    fh=figure(1);
    figure(fh);
    msg=['iteration: ' num2str(i) ' ,Objective Value: ' num2str(fun_val(i))];
    set(fh,'name',msg);imshow(max(x,[],3),[]);
  end
  
  if count >=5
    fun_val(i+1:end)=[];
    break;
  end
  
end

if ~exist('x0', 'var');
	x0 = Crop(y, x);
end

function Q=cost(y,Bop,f,b)

Hf=Bop(f)+b;
Hf(Hf<0)=0;
data_term=Hf(:)-y(:).*log(Hf(:));%0*log(0)=0 by convention. Matlab returns 
%NAN so we have to fix this. 
data_term(isnan(data_term))=0; 
data_term(isinf(data_term))=0;
Q=sum(data_term);

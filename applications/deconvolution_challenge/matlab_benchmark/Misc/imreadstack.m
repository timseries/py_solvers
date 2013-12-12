function f=imreadstack(imname)
% An extension of Matlab's function imread which reads at once an image-
% stack saved in a single file such as .tif


info = imfinfo(imname);
num_images = numel(info);
f=zeros(info(1).Height,info(1).Width,num_images);

for k = 1:num_images
    f(:,:,k) =imread(imname, k);
end
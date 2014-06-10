test_signal = y;

int_x_length = 4;
int_y_length = 4;
int_z_length = 4;
min_var = Inf;
min_mean = Inf;
max_patch_mean = Inf;
max_patch_location = 1;
patch_buffer_size = 200;
min_patches=cell(patch_buffer_size,1);
for x_index = 1:int_x_length:size(test_signal,1)
    for y_index = 1:int_y_length:size(test_signal,2)
        for z_index = 1:int_z_length:size(test_signal,3)
            black_patch_candidate = test_signal(x_index:x_index+int_x_length-1,y_index:y_index+int_y_length-1,z_index:z_index+int_z_length-1);
            min_var_candidate = var(double(black_patch_candidate(:)));
            min_mean_candidate = mean(black_patch_candidate(:));
%            if min_var_candidate < min_var
%            if min_mean_candidate < min_mean
            if min_mean_candidate < max_patch_mean %add to min_patches
                min_patches{max_patch_location}=black_patch_candidate;
%update the max patch mean value and location for the next one
                max_patch_mean = -Inf;
                for patch_index = 1:patch_buffer_size
                    if isempty(min_patches{patch_index})
                        max_patch_location = patch_index;
                        max_patch_mean = Inf;
                        break;
                    else
                        max_patch_mean_candidate = mean(min_patches{patch_index}(:));
                        if max_patch_mean_candidate > max_patch_mean
                            max_patch_location = patch_index;
                            max_patch_mean = max_patch_mean_candidate;
                        end
                    end
                end
            end
        end    
    end
end    
min_var
min_mean

%in case we didn't fill up the whole buffer, start fromt the first nonempty patch
first_patch_index = 1
while first_patch_index < patch_buffer_size
    if ~isempty(min_patches{first_patch_index})
        break;
    else
        first_patch_index = first_patch_index + 1;
    end
end    

min_vars=zeros(patch_buffer_size-first_patch_index+1,1);
min_means=zeros(patch_buffer_size-first_patch_index+1,1);
for patch_index = 1:(patch_buffer_size-first_patch_index+1)
    min_vars(patch_index) = var(double(min_patches{patch_index+first_patch_index-1}(:)));
    min_means(patch_index) = mean(min_patches{patch_index+first_patch_index-1}(:));
end    

vec_size = int_x_length*int_y_length*int_z_length;
var_vector = zeros(vec_size*patch_buffer_size,1);
var_pos = 1;
for patch_index = 1:patch_buffer_size
    var_vector(var_pos:var_pos+vec_size-1)=min_patches{patch_index}(:);
    var_pos = var_pos + vec_size;
end
%print stuff
min_vars
min_means
mean(min_vars)
b_est = mean(min_means)+1.5

sigma_est = sqrt(var(double(var_vector)))
k_est = max(y(:)) - 1.5*sigma_est - 1.5*b_est - b_est

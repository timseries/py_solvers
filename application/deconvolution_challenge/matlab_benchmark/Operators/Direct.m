function g = Direct(h, f)
	K = size(h);
	N = size(f);
	m = min(K, N);
	M = max(K, N);
% 	L = 0;
	L = mod(M, 2); % Perform the FFTs with even-length data (saves roughly 10 % of computation time)
% 	tic();
        disp(num2str(M+L));
        Hxhat_mat=fftn(h, M+L) .* fftn(f, M+L);
        xhat_mat=fftn(f, M+L);
	g = ifftn(fftn(h, M+L) .* fftn(f, M+L));
        x_mat=f;
% 	toc();
	v = colonvec(m, M);
%        v{1}
%        v{2}
%        v{3}
	g = g(v{:});
end
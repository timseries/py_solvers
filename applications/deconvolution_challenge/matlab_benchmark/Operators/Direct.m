function g = Direct(h, f)
	K = size(h);
	N = size(f);
	m = min(K, N);
	M = max(K, N);
% 	L = 0;
	L = mod(M, 2); % Perform the FFTs with even-length data (saves roughly 10 % of computation time)
% 	tic();
	g = ifftn(fftn(h, M+L) .* fftn(f, M+L));
% 	toc();
	v = colonvec(m, M);
	g = g(v{:});
end
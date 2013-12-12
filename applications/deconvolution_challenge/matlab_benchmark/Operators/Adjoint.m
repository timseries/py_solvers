function f = Adjoint(h, g)
	K = size(h);
	L = size(g);
	N = min(L+K-1, 2*L-1);
	f = zeros(N);
	v = colonvec(N-L+1, N);
	f(v{:}) = g;
% 	M = 0;
	M = mod(N, 2); % Perform the FFTs with even-length data (saves roughly 10 % of computation time)
% 	tic();
	f = ifftn(conj(fftn(h, N+M)) .* fftn(f, N+M));
% 	toc();
	v = colonvec(1, N-max(K-L, 0));
	f = f(v{:});
end
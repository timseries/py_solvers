function metric = FourierMetricsConstructor(N, K)
	% metric = FourierMetricsConstructor(N, K)
	%
	% Constructor.
	% Essentially precomputes the support of the Fourier shells.
	%
	% N: data size.
	% K: number of bins.
	%
	% metric: Fourier-metrics structure.
	%
	% (c) Cedric Vonesch, Biomedical Imaging Group, EPFL

	metric.K = K;
	D = numel(N);
	n = cell(D, 1);
	for d = 1:D
		n{d} = ifftshift(floor(-(N(d)-1)/2):floor((N(d)-1)/2));
	end
	[x{1:D}] = ndgrid(n{:});
	r = 0;
	for d = 1:D
		r = r + (2*x{d}/N(d)).^2;
	end
	r = sqrt(r);
	metric.i = cell(K, 1);
	metric.w = zeros(K, 1);
	epsilon = 1e-6;
	for k = 1:K
		metric.i{k} = find((k-1)/K-epsilon < r & r <= k/K);
		metric.w(k) = numel(metric.i(k));
	end
end

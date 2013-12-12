function c = FourierShellCorrelation(metric, fhat, ghat)
	% c = FourierShellCorrelation(metric, fhat, ghat)
	%
	% FSC between fhat and ghat.
	%
	% metric: Fourier-metrics structure.
	% fhat, ghat: DFTs of the data.
	%
	% c: FSC.
	%
	% (c) Cedric Vonesch, Biomedical Imaging Group, EPFL

	c = zeros(metric.K, 1);
	for k = 1:metric.K
		ftmp = fhat(metric.i{k});
		gtmp = ghat(metric.i{k});
		c(k) = ftmp' * gtmp / norm(ftmp) / norm(gtmp);
	end
end

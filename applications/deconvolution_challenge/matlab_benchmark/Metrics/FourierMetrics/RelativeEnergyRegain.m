function G = RelativeEnergyRegain(metric, Itld, Otld)
	% G = metric.RelativeEnergyRegain(Itld, Otld)
	%
	% RER of Itld with respect to Otld.
	%
	% Reference: Heintzmann. Estimating missing information by maximum likelihood
	% deconvolution. Micron 38 (2007) 136-144.
	%
	% metric: Fourier-metrics structure.
	% Otld: DFT of the ground truth.
	% Itld: DFT of the estimate.
	%
	% G: RER.
	%
	% (c) Cedric Vonesch, Biomedical Imaging Group, EPFL

	DeltaEbar = Itld - Otld;
	DeltaEbar = conj(DeltaEbar) .* DeltaEbar;

	Ebar = conj(Otld) .* Otld;

	G = zeros(metric.K, 1);
	for k = 1:metric.K
		G(k) = sum(Ebar(metric.i{k}));
		G(k) = (G(k) - sum(DeltaEbar(metric.i{k}))) / G(k);
	end
end

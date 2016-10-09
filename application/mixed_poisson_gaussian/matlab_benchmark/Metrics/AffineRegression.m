function fn = AffineRegression(f, f0)
	M = [norm(f(:))^2, sum(f(:)); sum(f(:)), numel(f)];
	c = M \ [f0(:).' * f(:); sum(f0(:))];
	fn = c(1) * f + c(2);

	% Validation: the output of AffineRegression(1:10, 1+2*(1:10)) should be [2; 1]
% 	disp(c);
end
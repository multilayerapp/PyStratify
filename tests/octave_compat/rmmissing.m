function B = rmmissing(A)
% Octave shim for MATLAB rmmissing on numeric arrays:
% row vector -> drop NaN elements; otherwise drop rows containing NaN.
if isrow(A)
  B = A(~isnan(A));
else
  B = A(~any(isnan(A), 2), :);
end
end

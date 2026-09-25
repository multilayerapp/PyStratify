function j = sbesselj( nu, z )
%SBESSELJ calculates spherical Bessel function of the first kind
% -------------------------------------------------------------------------
%% INPUT:
% -------------------------------------------------------------------------
% nu - the order of spherical Bessel function
% z  - argument
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% j - spherical Bessel function of the first kind
% -------------------------------------------------------------------------
%% COPYRIGHT
% -------------------------------------------------------------------------
% Copyright 2020 Ilia Rasskazov, University of Rochester
% -------------------------------------------------------------------------
% Author:        Ilia Rasskazov, irasskaz@ur.rochester.edu
% -------------------------------------------------------------------------
% Organization:  The Institute of Optics, University of Rochester
%                http://www.hajim.rochester.edu/optics/
% -------------------------------------------------------------------------
%% CALCULATING j
% -------------------------------------------------------------------------
if z == 0
    j = zeros(1,numel(nu));
    j(1,1) = 1;
else
    j = sqrt( pi./(2*z) ).*besselj( nu+0.5, z );
end
% -------------------------------------------------------------------------
end
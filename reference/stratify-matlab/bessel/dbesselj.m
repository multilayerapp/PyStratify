function dJ = dbesselj( nu, z )
%DBESSELJ calculates first derivative of the Bessel function of the first kind
% -------------------------------------------------------------------------
%% INPUT:
% -------------------------------------------------------------------------
% nu - the order of the Bessel function
% z  - argument
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% dJ - first derivative of the Bessel function of the first kind
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
%% CALCULATING dJ
% -------------------------------------------------------------------------
dJ = 0.5*(besselj(nu-1,z) - besselj(nu+1,z));
% -------------------------------------------------------------------------
end
function dh = dsbesselh( nu, k, z )
%DSBESSELH calculates first derivative of 
%                       the spherical Bessel function of the third kind
% -------------------------------------------------------------------------
%% INPUT:
% -------------------------------------------------------------------------
% nu - the order of the Bessel function
% k  - 1 or 2, default = 1
% z  - argument
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% dh  - first derivative of the spherical Bessel function of the third kind
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
%% CHECKING INPUT
% -------------------------------------------------------------------------
if nargin == 2
    z = k;
    k = 1;
end
% -------------------------------------------------------------------------
%% CALCULATING dh
% -------------------------------------------------------------------------
dh = 0.5.*( sbesselh( nu-1, k, z ) - ...
            sbesselh( nu+1, k, z ) - ...
            sbesselh( nu,   k, z )./z );
% -------------------------------------------------------------------------
end
function dxi = dricbesh( nu, k, z )
%DRICBESH calculates first derivative of 
%                      the Riccati-Bessel function of the third kind
% -------------------------------------------------------------------------
%% INPUT:
% -------------------------------------------------------------------------
% nu - the order of the Bessel function
% k  - 1 or 2, default = 1
% z  - argument
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% dxi - first derivative of the Riccati-Bessel function of the third kind
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
%% CALCULATING dxi
% -------------------------------------------------------------------------
dxi = 0.5*sqrt( pi/2./z ).* ... 
        besselh( nu+0.5, k, z ) + sqrt( pi*z/2 ).*dbesselh( nu+0.5, k, z );
% -------------------------------------------------------------------------
end
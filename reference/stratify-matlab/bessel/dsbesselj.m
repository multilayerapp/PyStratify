function dj = dsbesselj( nu, z )
%DSBESSELJ calculates first derivative of 
%                       the spherical Bessel function of the first kind
% -------------------------------------------------------------------------
%% INPUT:
% -------------------------------------------------------------------------
% nu - the order of the Bessel function
% z  - argument
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% dj - first derivative of the spherical Bessel function of the first kind
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
%% CALCULATING dj
% -------------------------------------------------------------------------
dj = - sbesselj( nu+1, z ) + ( nu./z ).*sbesselj( nu, z );
% -------------------------------------------------------------------------
end
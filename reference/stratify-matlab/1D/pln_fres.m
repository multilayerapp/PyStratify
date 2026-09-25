function [ rs, rp, ts, tp ] = pln_fres( n1, n2, theta1 )
%% fres_coef calculates Fresnel reflection and transmission coefficients 
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% n1     - complex refractive index of the medium where wave comes from
% n2     - complex refractive index of the medium where wave comes to     
% theta1 - angle of incidence
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% rs     - reflection coefficent for s-polarized (TE) mode
% rp     - reflection coefficent for p-polarized (TM) mode
% ts     - transmission coefficent for s-polarized (TE) mode
% tp     - transmission coefficent for p-polarized (TM) mode
% -------------------------------------------------------------------------
%% REFERENCE
% -------------------------------------------------------------------------
% This formalism can be found in almost every textbook on optics
% -------------------------------------------------------------------------
%% COPYRIGHT
% -------------------------------------------------------------------------
% Copyright 2021 Ilia Rasskazov, University of Rochester
% -------------------------------------------------------------------------
% Author:        Ilia Rasskazov, irasskaz@ur.rochester.edu
% -------------------------------------------------------------------------
% Organization:  The Institute of Optics, University of Rochester
%                http://www.hajim.rochester.edu/optics/
% -------------------------------------------------------------------------
%% CALCULATING COEFFICIENTS
% -------------------------------------------------------------------------
theta2 = asin( n1./n2.*sin( theta1 ) );                                     % Snell's law
% -------------------------------------------------------------------------
rs = ( n1.*cos( theta1 ) - n2.*cos( theta2 ) )./...
     ( n1.*cos( theta1 ) + n2.*cos( theta2 ) );
rp = ( n1.*cos( theta2 ) - n2.*cos( theta1 ) )./...
     ( n1.*cos( theta2 ) + n2.*cos( theta1 ) );
% -------------------------------------------------------------------------
ts =            2*n1.*cos( theta1 )./...
     ( n1.*cos( theta1 ) + n2.*cos( theta2 ) );
tp =            2*n1.*cos( theta1 )./...
     ( n1.*cos( theta2 ) + n2.*cos( theta1 ) );
% -------------------------------------------------------------------------
end
function [A, T, R] = pln_get_ATR( n, lam, d, theta0 )
%% pln_get_ATR calculates absorptance, transmittance and reflectance
%                                               for multilayered structure
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% n      - matrix for complex refractive indices of layers
%          rows correspond to layers
%          columns correspond to wavenumbers
% lam    - wavelength, scalar, nanometers
% d      - thicknesses of each layer, nanometers
% theta0 - angle of incidence
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% A      - absorptance
% T      - transmittance
% R      - reflectance
% -------------------------------------------------------------------------
%% REFERENCE
% -------------------------------------------------------------------------
% See for details:
% K. Ohta and H. Ishida, 
% Matrix formalism for calculation of electric field intensity of light 
%                                       in stratified multilayered films 
% Appl. Opt. 29(13), 1952–1959 (1990); 10.1364/AO.29.001952
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
theta_full = zeros( size( n, 1 ), size( n, 2 ) );                      
theta_full(1,:) = theta0;
% -------------------------------------------------------------------------
for i = 2:size( n, 1 )
    theta_full(i,:) = asin( n(i-1,:)./n(i,:).*sin( theta_full(i-1,:) ) );   % Snell's law
end
theta_last = theta_full(end,:);
%
[ rs, rp, ts, tp ] = pln_abcd( n, lam, d, theta0 );

T = 0.5*(real(n(end,:).*cos(theta_last)./n(1,:)./cos(theta0)).'.*abs(ts).^2 + ...
         real(conj(n(end,:)).*cos(theta_last)./conj(n(1,:))./cos(theta0)).'.*abs(tp).^2);
R = (abs(rs).^2 + abs(rp).^2)/2;
A = 1 - T - R;

end
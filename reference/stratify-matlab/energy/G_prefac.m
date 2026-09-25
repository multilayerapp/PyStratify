function G = G_prefac( mat, lam )
%G_PREFAC calculates pre-factors G_e and G_m entering the expression for 
%          electromagnetic energy radial density
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% mat - string array with materials
% lam - vacuum wavelength of incident illumination, meters
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% G - structure with electric (e) and magnetic (m) G for each shell
% -------------------------------------------------------------------------
%% REFERENCES
% -------------------------------------------------------------------------
% Constants for Drude fitting of Au, Ag and Al:
% [1] Blaber, M. G.; Arnold, M. D.; Ford, M. J.
%       Search for the ideal plasmonic nanoshell: 
%       the effects of surface scattering and alternatives to gold and silver
%       Journal of Physical Chemistry C 2009, 113(8), 3041-3045
%       https://pubs.acs.org/doi/10.1021/jp810808h
% [2] Ordal, M. A.; Bell, Robert J.; Alexander, R. W.; Long, L. L.; Querry, M. R.
%       Optical properties of fourteen metals in the infrared and far infrared: 
%       Al, Co, Cu, Au, Fe, Pb, Mo, Ni, Pd, Pt, Ag, Ti, V, and W
%       Applied Optics, 1985, 24(24), 4493-4499
%       https://doi.org/10.1364/AO.24.004493
% -------------------------------------------------------------------------
%% COPYRIGHT
% -------------------------------------------------------------------------
% Copyright 2020 Ilia Rasskazov, University of Rochester
% -------------------------------------------------------------------------
% Author:        Ilia Rasskazov
%                irasskaz@ur.rochester.edu
% -------------------------------------------------------------------------
% Organization:  The Institute of Optics, University of Rochester
%                http://www.hajim.rochester.edu/optics/
% -------------------------------------------------------------------------
%% ALLOCATING USEFUL QUANTITIES
% -------------------------------------------------------------------------
G.e = zeros( numel( mat ), numel( lam ) );
G.m = zeros( numel( mat ), numel( lam ) );
% -------------------------------------------------------------------------
%% CALCULATING PRE-FACTORS
% -------------------------------------------------------------------------      
for i = 1 : numel(mat)
% -------------------------------------------------------------------------
    switch mat(i)
        case 'Au_Ord'
            lam_g  = 46436.02734e-9;
            lam_p  = 137.36e-9;
            eps    = 1 - 1/(lam_p/lam*(lam_p/lam + 1i.*lam_p/lam_g));
            G.e(i) = real(eps) + 2*imag(eps)*lam_g/lam_p;
            G.m(i) = 1;
        case 'Ag_Ord'
            lam_g  = 68880.10723e-9;
            lam_p  = 137.56e-9;
            eps    = 1 - 1/(lam_p/lam*(lam_p/lam + 1i.*lam_p/lam_g));
            G.e(i) = real(eps) + 2*imag(eps)*lam_g/lam_p;
            G.m(i) = 1;
        case 'Al_Ord'
            lam_g  = 15156.99181e-9;
            lam_p  = 84.05708e-9;
            eps    = 1 - 1/(lam_p/lam*(lam_p/lam + 1i.*lam_p/lam_g));
            G.e(i) = real(eps) + 2*imag(eps)*lam_g/lam_p;
            G.m(i) = 1;
        case 'Au_Blb'
            lam_g  = 67382.71359e-9;
            lam_p  = 144.93e-9;
            eps    = 1 - 1/(lam_p/lam*(lam_p/lam + 1i.*lam_p/lam_g));
            G.e(i) = real(eps) + 2*imag(eps)*lam_g/lam_p;
            G.m(i) = 1;
        case 'Ag_Blb'
            lam_g  = 54379.03202e-9;
            lam_p  = 129.15e-9;
            eps    = 1 - 1/(lam_p/lam*(lam_p/lam + 1i.*lam_p/lam_g));
            G.e(i) = real(eps) + 2*imag(eps)*lam_g/lam_p;
            G.m(i) = 1;
        case 'Al_Blb'
            lam_g  = 2071.92836e-9;
            lam_p  = 81.03542e-9;
            eps    = 1 - 1/(lam_p/lam*(lam_p/lam + 1i.*lam_p/lam_g));
            G.e(i) = real(eps) + 2*imag(eps)*lam_g/lam_p;
            G.m(i) = 1;
        case 'Si_AS'
            G.e(i) = 3.9765;
            G.m(i) = 1;
        case 'SiO2'
            G.e(i) = 2.1025;
            G.m(i) = 1;
        case 'Al2O3'
            G.e(i) = 3.0976;
            G.m(i) = 1;
        case 'ZnO'
            G.e(i) = 4;
            G.m(i) = 1;
        case 'NaYF4'
            G.e(i) = 2.1609;
            G.m(i) = 1;
        case 'BaTiO3'
            G.e(i) = 6.8121;
            G.m(i) = 1;
        case 'H2O'
            G.e(i) = 1.78;
            G.m(i) = 1;
        case 'TiO2'
            G.e(i) = 5.913 + 0.2441/((lam*1e6)^2 - 0.0803);
            G.m(i) = 1;
        case 'vac'
            G.e(i) = 1;
            G.m(i) = 1;
        otherwise
            error('Material for G-prefactor is not found');
    end
% -------------------------------------------------------------------------    
end
% -------------------------------------------------------------------------
end
function n_out = el_fr_pth( lam, n_in, rad, mat)
%ELEC_FREE_PATH calculates electron free path corrected refractive index
%                for thin shells
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% lam  - wavelength, meters
% n_in - tabulated refractive index for a bulk material
% rad  - inner and outer radii of the shell, meters
% mat  - material: Ag, Au or Al from Ordal (Ord) or Blaber (Blb) database
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% n_out - corrected refractive index
% -------------------------------------------------------------------------
%% REFERENCES
% -------------------------------------------------------------------------
% electron free path definition for thin shells:
% Moroz, A. 
% Electron mean free path in a spherical shell geometry
% Journal of Physical Chemistry C 2008, 112(29), 10641-10652
% https://pubs.acs.org/doi/10.1021/jp8010074
% -------------------------------------------------------------------------
% Fermi velocities:
% Kreibig, U.; Vollmer, M.
% Optical Properties of Metal Clusters
% http://link.springer.com/10.1007/978-3-662-09109-8
% -------------------------------------------------------------------------
% Constants for Drude fitting:
% [1] Blaber, M. G.; Arnold, M. D.; Ford, M. J.
%       Search for the ideal plasmonic nanoshell: 
%       the effects of surface scattering and alternatives to gold and silver
%       Journal of Physical Chemistry C 2009, 113(8), 3041-3045
%       https://pubs.acs.org/doi/10.1021/jp810808h
% [2] Ordal, M. A.; Bell, Robert J.; Alexander, R. W.; Long, L. L.; Querry, M. R.
%       Optical properties of fourteen metals in the infrared and far infrared: 
%       Al, Co, Cu, Au, Fe, Pb, Mo, Ni, Pd, Pt, Ag, Ti, V, and W
%       Applied Optics?1985, 24(24),?4493-4499
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
%% CALCULATING ELECTRON FREE PATH
% -------------------------------------------------------------------------
if numel(rad) == 1                                                          % for core
    L = 4/3*rad;
else                                                                        % for any other shell
    L = 4/3*(rad(2)^3-rad(1)^3)/(rad(1)^2+rad(2)^2);    
end
% -------------------------------------------------------------------------
switch mat                                                                  % picking the material 
    case 'Au_Ord'
        vf_c  = 0.014/3;                                                    % Fermi velocity divided by the speed of light
        lam_p = 137.36e-9;                                                  % plasma wavelength, nm   
        gamma_p = 0.0267/9.026;
    case 'Ag_Ord'
        vf_c  = 0.0139/3;
        lam_p = 137.56e-9;
        gamma_p = 0.018/9.013;
    case 'Al_Ord' 
        vf_c  = 0.0206/3;                                                   % Fermi velocity divided by the speed of light
        lam_p = 84.05708e-9;                                                % plasma wavelength, nm   
        gamma_p = 0.0818/14.75;
     case 'Au_Blb' 
        vf_c  = 0.014/3;                                                    % Fermi velocity divided by the speed of light
        lam_p = 144.93e-9;                                                  % plasma wavelength, nm   
        gamma_p = 0.0184/8.55;
    case 'Ag_Blb'
        vf_c  = 0.0139/3;
        lam_p = 129.15e-9;
        gamma_p = 0.0228/9.6;
   case 'Al_Blb'
        vf_c  = 0.0206/3;
        lam_p = 81.03542e-9;
        gamma_p = 0.5984/15.3;
    otherwise
        error('Material for electron free-path correction is not found');
end
% -------------------------------------------------------------------------
%% CONSTRUCTING THE OUTPUT
% -------------------------------------------------------------------------
eps_bulk = n_in.^2;
llp = lam_p./lam;
gamma_s = gamma_p + 0.5*vf_c*lam_p/(L*pi);
n_out = sqrt( eps_bulk + ...
              1./( llp.*(llp + 1i*gamma_p) ) - ...
              1./( llp.*(llp + 1i*gamma_s) )); 
% -------------------------------------------------------------------------
end
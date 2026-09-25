function l_max = l_conv( rad, ref, lam, fld )
%l_conv calculates l_max required for truncation
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% rad - outer radii of the sphere
% ref - refractive index of the host medium
% lam - vacuum wavelength (can be array with all wavelengths)
% fld - regime: 'far' or 'near' field
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% l_max - truncation l calculated via Wiscombe criterion [1] for far field
%           or with amended criterion for a near field [2]
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
%% REFERENCES
% -------------------------------------------------------------------------
% [1] W. J. Wiscombe, Improved Mie scattering algorithms,
%   Appl. Opt. 19(9), 1505–1509 (1980) 
%   https://doi.org/10.1364/AO.19.001505
% [2]﻿J. R. Allardice and E. C. Le Ru
%   Convergence of Mie theory series: 
%   criteria for far-field and near-field properties,
%   Appl. Opt. 53(31), 7224–7229 (2014)
%   https://doi.org/10.1364/AO.53.007224
% -------------------------------------------------------------------------
%% CALCULATING SIZE PARAMETER
% -------------------------------------------------------------------------
x = 2*pi*rad*ref/min(lam,[],'all');
% -------------------------------------------------------------------------
%% CONSTRUCTING OUTPUT
% -------------------------------------------------------------------------
switch fld
    case 'near'
        l_max = x + 11*x^(1/3) + 1;
    case 'far'
        if x <= 8
            l_max = x + 4*x^(1/3) + 1;
        elseif x < 4200
            l_max = x + 4.05*x^(1/3) + 2;
        else
            error('too large sphere. exiting');
        end
end
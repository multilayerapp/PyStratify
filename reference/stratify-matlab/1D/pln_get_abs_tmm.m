function  A = pln_get_abs_tmm( IR, wn, NA )
%% GET_ABS_FRES calculates reflection and transmission coefficients 
%               for multilayered structure via transfer matrix method
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% n_full - matrix for complex refractive indices of full system
%          rows correspond to layers
%          columns correspond to wavenumbers
% n_bare - matrix for complex refractive indices of system WITHOUT a sample
%          rows correspond to layers
%          columns correspond to wavenumbers
% wn     - wavenumber, scalar, inverse microns
% d      - thicknesses of each layer, microns
% NA     - numerical apertures for illumination and detection
% geom   - 'transmission'/'transflection', the optical setup
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
%% OUTPUT
% -------------------------------------------------------------------------
% A      - apparent absorbance
% -------------------------------------------------------------------------
%% ALLOCATING OUTPUT
% -------------------------------------------------------------------------
A = zeros(100,numel(wn));
% -------------------------------------------------------------------------
%% CONVERTING
% -------------------------------------------------------------------------
wn = wn * 1e-4;                                                             % from cm^-1 to microns^-1  
% -------------------------------------------------------------------------
%% CALCULATING ABSORBANCE FOR EACH ANGLE
% -------------------------------------------------------------------------
switch IR.geom
% -------------------------------------------------------------------------    
    case 'transmission'
% -------------------------------------------------------------------------
        theta = linspace(asin(NA.cob),asin(NA.c));
        parfor i = 1:100
            [ ~, ~, Ts_full, Tp_full ] = ...
                pln_abcd( IR.n_full, wn, IR.d_full, theta(i) );
            [ ~, ~, Ts_bare, Tp_bare ] = ...
                pln_abcd( IR.n_bare, wn, IR.d_bare, theta(i) );
            A(i,:) = log10( (abs( Ts_bare ).^2 + abs( Tp_bare ).^2)./ ...
                            (abs( Ts_full ).^2 + abs( Tp_full ).^2) );
        end
% -------------------------------------------------------------------------
    case 'transflection'
% -------------------------------------------------------------------------        
        theta = linspace(asin(NA.cob),asin(NA.c)); 
        parfor i = 1:100        
            [ Rs_full, Rp_full, ~, ~ ] = ...
                pln_abcd( IR.n_full, wn, IR.d_full, theta(i) );
            [ Rs_bare, Rp_bare, ~, ~ ] = ...
                pln_abcd( IR.n_bare, wn, IR.d_bare, theta(i) );
            A(i,:) = log10( (abs( Rs_bare ).^2 + abs( Rp_bare ).^2)./ ...
                            (abs( Rs_full ).^2 + abs( Rp_full ).^2) );
        end
% -------------------------------------------------------------------------
    otherwise
% -------------------------------------------------------------------------        
        error('incorrect setup of the microscope')
% -------------------------------------------------------------------------
end
% -------------------------------------------------------------------------
%% OUTPUT AND AVERAGING ALONG ANGLES
% -------------------------------------------------------------------------
A = mean( A );
% -------------------------------------------------------------------------
end
function [ sc, ab, ex ] = crs_sec( rad, lam, nh, l, T )
%CRS_SEC calculates scattering, absorption and extinction 
% of a general multilayered spherical particle 
% in a general host (lossless, lossy or with gain)
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% rad - outer radii for each layer of the sphere
% lam - vacuum wavelength
% nh  - refractive index of the host medium
% l   - multipoles (array or scalar)
% T   - transfer matrix calculated from "t_mat.m"
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% sc - scattering
% ab - absorption
% ex - extinction
%       key to structures:  cs - cross-section
%                           ef - efficiency
%                           e  - electric (TM mode)
%                           m  - magnetic (TE mode)
%                           em - e+m
%                           l  - multipole decomposition of the respective quantity
% -------------------------------------------------------------------------
%% ALLOCATING USEFUL QUANTITIES
% -------------------------------------------------------------------------
% wavevector:
k    = 2*pi*nh/ lam;
% coefficients, notice real part of k necessary for lossy or gain host
cf1  = 2 * pi / real(k);                                                   
cf2  = 2 * pi / real(k)^2;
% geometric cross-section
gcs  = pi * rad(end)^2;                                                     
% -------------------------------------------------------------------------
%% CALCULATING EXPANSION COEFFICIENTS
% -------------------------------------------------------------------------
a = -(squeeze(T.te(:,end,2,1))./squeeze(T.te(:,end,1,1))).';
b = -(squeeze(T.tm(:,end,2,1))./squeeze(T.tm(:,end,1,1))).';
% -------------------------------------------------------------------------
%% CONSTRUCTING OUTPUT
% -------------------------------------------------------------------------
% cross-sections for each multipole l
sc.csel  = cf2.*( 2*l + 1 ).*abs(a).^2;                                     
sc.csml  = cf2.*( 2*l + 1 ).*abs(b).^2;
sc.cseml = sc.csml + sc.csel;
ex.csel  = cf1.*( 2*l + 1 ).*real(a./k);
ex.csml  = cf1.*( 2*l + 1 ).*real(b./k);
ex.cseml = ex.csel + ex.csml;
ab.csel  = ex.csel - sc.csel;
ab.csml  = ex.csml - sc.csml;
ab.cseml = ab.csel + ab.csml;
% -------------------------------------------------------------------------
% total cross-sections
sc.cse  = sum( sc.csel );                                                   
sc.csm  = sum( sc.csml );
sc.csem = sum( sc.cseml );
ex.cse  = sum( ex.csel );
ex.csm  = sum( ex.csml );
ex.csem = sum( ex.cseml );
ab.cse  = ex.cse - sc.cse;
ab.csm  = ex.csm - sc.csm;
ab.csem = ab.cse + ab.csm;
% -------------------------------------------------------------------------
% scattering, extinction and absorption efficiencies for each multipole l
sc.efel  = sc.csel / gcs;                                                  
sc.efml  = sc.csml / gcs;
sc.efeml = sc.cseml / gcs;
ex.efel  = ex.csel / gcs;
ex.efml  = ex.csml / gcs;
ex.efeml = ex.cseml / gcs;
ab.efel  = ab.csel / gcs;
ab.efml  = ab.csml / gcs;
ab.efeml = ab.cseml / gcs;
% -------------------------------------------------------------------------
% total efficiencies
sc.efe  = sc.cse / gcs;                                                    
sc.efm  = sc.csm / gcs;
sc.efem = sc.csem / gcs;
ex.efe  = ex.cse / gcs;
ex.efm  = ex.csm / gcs;
ex.efem = ex.csem / gcs;
ab.efe  = ab.cse / gcs;
ab.efm  = ab.csm / gcs;
ab.efem = ab.csem / gcs;
% -------------------------------------------------------------------------
end
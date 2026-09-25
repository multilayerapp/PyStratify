% This script calculates angle-resolved scattering pattern of a SiO2@Au
% core-shell sphere in a vacuum
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% setting up incident illumination and detection
lam = 750e-9;                                                               % wavelength
l   = 1:5;                                                                  % up to 5-th multipole
th = linspace(0,1.999*pi,101);                                              % theta for scattering
% -------------------------------------------------------------------------
% setting up refractive indices
load( 'Au_JC.mat' );                                                        % loading refractive indices
nsio2 = 1.45;                                                               % silica core
nau   = complex( interp1( Au_JC(:,1), Au_JC(:,2), lam, 'spline' ),...       % interpolating tabulated data for Au
                 interp1( Au_JC(:,1), Au_JC(:,3), lam, 'spline') );          
nh    = 1;                                                                  % vacuum host
% -------------------------------------------------------------------------
% setting up particle
rad = [50,70]*1e-9;                                                         % radii of shells, meters
mu  = ones(1,3);                                                            % permeabilities, including the host medium     
nb = [nsio2, nau, nh];                                                      % refractive indices
% -------------------------------------------------------------------------
%% CALCULATING S
% -------------------------------------------------------------------------
T = t_mat( rad, nb, mu, lam, l );                                           % transfer matrices
S = far_fld(l, T, th);                                                      % angle-resolved scattering
% -------------------------------------------------------------------------
%% PLOTTING RESULTS
% -------------------------------------------------------------------------
polarplot(th,abs(S.per).^2,th,abs(S.par).^2,'LineWidth',2);
title('$|S(\theta)|^2$','Interpreter','latex'); 
legend('$|S_\perp(\theta)|^2$','$|S_\parallel(\theta)|^2$','Interpreter','latex');
% -------------------------------------------------------------------------
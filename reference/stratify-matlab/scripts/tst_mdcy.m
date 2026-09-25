% this script calculates decay rates of the electric and magnetic dipole 
% emitters near Ag sphere and reproduces Fig.4(c)(d) of:
% Schmidt MK, Esteban R, Sáenz JJ, Suárez-Lacalle I, Mackowski S, Aizpurua J. 
% Dielectric antennas - a suitable platform for controlling magnetic dipolar emission. 
% Opt Express. 2012;20(13):13636. doi:10.1364/OE.20.013636
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% setting up incident illumination
lam = linspace( 300, 700, 401 )*1e-9;                                       % wavelength
l   = 1:100;                                                                % up to 100-th multipole
% -------------------------------------------------------------------------
% setting up refractive indices
load( 'Ag_JC.mat' );                                                        % loading refractive index
nag = complex( interp1( Ag_JC(:,1), Ag_JC(:,2), lam ), ...
               interp1( Ag_JC(:,1), Ag_JC(:,3), lam) );
nh    = 1;                                                                  % air host
% -------------------------------------------------------------------------
% setting up particle
rad = 50e-9;                                                                % radii of sphere
mu  = ones(1,2);                                                            % permeabilities, including the host medium
% -------------------------------------------------------------------------
% setting up properties of the emitter and parameters for decay rates
rdip = 60e-9;                                                               % distance from a sphere center
rint = 200;                                                                 % points for integrals in I_abs function
% -------------------------------------------------------------------------
% preallocating data for decay rates
gre = zeros( numel( lam ), 2 ); grm = zeros( numel( lam ), 2 );             % radiative 
gnre = zeros( numel( lam ), 2 ); gnrm = zeros( numel( lam ), 2 );           % nonradiative
% -------------------------------------------------------------------------
%% CALCULATING DECAY RATES
% -------------------------------------------------------------------------
for il = 1 : numel( lam )                                                   % loop over lambdas
    nb = [nag(il),nh];
    T = t_mat( rad, nb, mu, lam(il), l );                                   % transfer matrices
    [ gre(il,:), gnre(il,:) ] = ...
        edcy( rad, rdip, nb, mu, lam(il), l, rint, T, 'host' );        % electric dipole decay rates
    [ grm(il,:), gnrm(il,:) ] = ...
        mdcy( rad, rdip, nb, mu, lam(il), l, rint, T, 'host' );        % magnetic dipole decay rates
end
% -------------------------------------------------------------------------
%% PLOTTING RESULTS
% -------------------------------------------------------------------------
lam = lam*1e9;
% -------------------------------------------------------------------------
% Radiative decay rates
% -------------------------------------------------------------------------
figure();
semilogy( lam, gre(:,1),'b-',...
          lam, gre(:,2),'r-',...
          lam, grm(:,1), 'b--',...
          lam, grm(:,2), 'r--', ...
                        'LineWidth', 2 );
xlim([lam(1) lam(end)]);
legend('${\perp}$, ED', ...
       '${\parallel}$, ED', ...
       '${\perp}$, MD', ...
       '${\parallel}$, MD', ...
       'Interpreter','latex','FontSize',14);
title('Radiative Decay Rates','Interpreter','latex');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
% Nonradiative decay rates
% -------------------------------------------------------------------------
figure();
semilogy( lam, gre(:,1)+gnre(:,1),'b-',...
          lam, gre(:,2)+gnre(:,2),'r-',...
          lam, grm(:,1)+gnrm(:,1), 'b--',...
          lam, grm(:,2)+gnrm(:,2), 'r--', ...
                        'LineWidth', 2 );
xlim([lam(1) lam(end)]);
legend('${\perp}$, ED', ...
       '${\parallel}$, ED', ...
       '${\perp}$, MD', ...
       '${\parallel}$, MD', ...
       'Interpreter','latex','FontSize',14);
title('Total Decay Rates','Interpreter','latex');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
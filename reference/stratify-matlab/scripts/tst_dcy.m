% this script calculates decay rates of the dipole emitter near or inside 
% the shell of Au@SiO2 core-shell sphere
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% setting up incident illumination
lam = linspace( 300, 1000, 701 )*1e-9;                                      % wavelength
l   = 1:100;                                                                % up to 100-th multipole
% -------------------------------------------------------------------------
% setting up refractive indices
load( 'Au_JC.mat' );                                                        % loading refractive index
nsio2 = 1.45;                                                               % silica core
nau   = complex( interp1( Au_JC(:,1), Au_JC(:,2), lam, 'spline' ),...       % interpolating tabulated data for Au
                 interp1( Au_JC(:,1), Au_JC(:,3), lam, 'spline') );          
nh    = 1.33;                                                               % water host
% -------------------------------------------------------------------------
% setting up particle
rad = [50,70]*1e-9;                                                         % radii of shells, meters
mu  = ones(1,3);                                                            % permeabilities, including the host medium
% -------------------------------------------------------------------------
% setting up properties of the emitter and parameters for decay rates
rdip = linspace( 52, 150, 99 )*1e-9;                                        % distance from a sphere center
rint = 200;                                                                 % points for integrals in I_abs function
% -------------------------------------------------------------------------
%% CALCULATING DECAY RATES
% -------------------------------------------------------------------------
% distance-dependent at lam = 614nm
nb = [nau(315),nsio2,nh];                                                   % refractive indices
T = t_mat( rad, nb, mu, lam(315), l );                                      % transfer matrices
[ grdist, gnrdist ] = ...
    edcy( rad, rdip, nb, mu, lam(315), l, rint, T, 'host' );          % decay rates
% -------------------------------------------------------------------------
% wavelength-dependent at rdipl = 71nm distance from a sphere center
rdipl = 71e-9;
grlam = zeros( numel( lam ), 2 ); gnrlam = zeros( numel( lam ), 2 );
parfor il = 1 : numel( lam )                                                % loop over lambdas
    nb = [nau(il),nsio2,nh];                                                % refractive indices
    T = t_mat( rad, nb, mu, lam(il), l );                                   % transfer matrices
    [ grlam(il,:), gnrlam(il,:) ] = ...
        edcy( rad, rdipl, nb, mu, lam(il), l, rint, T, 'host' );      % decay rates
end
% -------------------------------------------------------------------------
%% PLOTTING RESULTS
% -------------------------------------------------------------------------
rdip = rdip*1e9; lam = lam*1e9;
% -------------------------------------------------------------------------
figure();
% -------------------------------------------------------------------------
subplot(1,2,1);
semilogy( rdip, grdist, rdip, gnrdist, '--', 'LineWidth', 2 );
xlim([rdip(1) rdip(end)]); xline(71,':k','LineWidth', 1);
legend('$\Gamma_{\rm rad}^{\perp}$', ...
       '$\Gamma_{\rm rad}^{\parallel}$', ...
       '$\Gamma_{\rm nrad}^{\perp}$', ...
       '$\Gamma_{\rm nrad}^{\parallel}$',...
       'Interpreter','latex');
title('Decay Rates, $\lambda=614$nm','Interpreter','latex');
xlabel('$r$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(1,2,2);
plot( lam, grlam, lam, gnrlam, '--',  'LineWidth', 2 );
xlim([lam(1) lam(end)]); xline(614,':k','LineWidth', 1);
legend('$\Gamma_{\rm rad}^{\perp}$', ...
       '$\Gamma_{\rm rad}^{\parallel}$', ...
       '$\Gamma_{\rm nrad}^{\perp}$', ...
       '$\Gamma_{\rm nrad}^{\parallel}$',...
       'Interpreter','latex');
title('Decay Rates, $r_d=71$nm','Interpreter','latex');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
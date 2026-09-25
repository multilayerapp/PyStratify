% this script calculates scattering, absorption and extinction efficiencies
% of SiO2@Au core-shell sphere and shows the importance of free electron 
% path effects which modify the permittivity of 5nm-thick Au shell
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% setting up incident illumination
lam = linspace( 600, 1000, 801 )*1e-9;                                      % wavelength
% -------------------------------------------------------------------------
% setting up refractive indices
load( 'Au_JC.mat' );                                                        % loading refractive indices
nsio2 = 1.45;                                                               % silica core
nau   = complex( interp1( Au_JC(:,1), Au_JC(:,2), lam, 'spline' ),...       % interpolating tabulated data for Au
                 interp1( Au_JC(:,1), Au_JC(:,3), lam, 'spline') );          
nh    = 1;                                                                  % vacuum host
% -------------------------------------------------------------------------
% setting up particle
rad = [50,55]*1e-9;                                                         % radii of shells, meters
mu  = ones(1,3);                                                            % permeabilities, including the host medium     
% -------------------------------------------------------------------------
% setting up output
abblk = zeros( numel( lam ), 1 ); abcor = zeros( numel( lam ), 1 );
exblk = zeros( numel( lam ), 1 ); excor = zeros( numel( lam ), 1 );
scblk = zeros( numel( lam ), 1 ); sccor = zeros( numel( lam ), 1 );
% -------------------------------------------------------------------------
%% CALCULATING SCATTERING, ABSORPTION AND EXTINCTION
% -------------------------------------------------------------------------
parfor i = 1 : numel(lam)
    l = 1 : l_conv( rad(end), nh, lam(i), 'far' );                          % defining l required for convergence
    nb = [nsio2, nau(i), nh];                                               % using bulk n for Au
    T = t_mat( rad, nb, mu, lam(i), l );                                    % transfer matrices
    [ sc, ab, ex ] = crs_sec( rad, lam(i), nh, l, T );                      % getting sc, ab, ex
    abblk(i) = ab.efem;  scblk(i) = sc.efem;  exblk(i) = ex.efem;           % writing output
    nauc = el_fr_pth( lam(i), nau(i), rad(1:2), 'Au_Ord' );                 % electron free path correction
    nc  = [ nsio2, nauc, nh ];                                              % using corrected n for Au
    T = t_mat( rad, nc, mu, lam(i), l );                                    % transfer matrices
    [ sc, ab, ex ] = crs_sec( rad, lam(i), nh, l, T );                      % getting sc, ab, ex
    abcor(i) = ab.efem;  sccor(i) = sc.efem;  excor(i) = ex.efem;           % writing output
end
% -------------------------------------------------------------------------
%% PLOTTING RESULTS
% -------------------------------------------------------------------------
lam = lam*1e9;
% -------------------------------------------------------------------------
figure();
% -------------------------------------------------------------------------
subplot(3,1,1); plot(lam,abblk,'--',lam,abcor,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(exblk,[],'all')]);
title('absorption'); legend('bulk','corrected');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(3,1,2); plot(lam,scblk,'--',lam,sccor,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(exblk,[],'all')]);
title('scattering'); legend('bulk','corrected');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(3,1,3); plot(lam,exblk,'--',lam,excor,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(exblk,[],'all')]);
title('extinction'); legend('bulk','corrected');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
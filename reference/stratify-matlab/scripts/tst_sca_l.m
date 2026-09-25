% this script calculates scattering, absorption and extinction efficiencies
% of Au@SiO2 core-shell sphere and shows the contribution of multipoles
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% setting up incident illumination
lam = linspace( 300, 1500, 1201 )*1e-9;                                     % wavelength
l   = 1:3;                                                                  % up to 3-rd multipole
% -------------------------------------------------------------------------
% setting up refractive indices
load( 'Au_JC.mat' );                                                        % loading refractive indices
nsio2 = 1.45;                                                               % silica shell
nau   = complex( interp1( Au_JC(:,1), Au_JC(:,2), lam, 'spline' ),...       % interpolating tabulated data for Au
                 interp1( Au_JC(:,1), Au_JC(:,3), lam, 'spline') );          
nh    = 1.33;                                                               % water host
% -------------------------------------------------------------------------
% setting up particle
rad = [110,120]*1e-9;                                                       % radii of shells, meters
mu  = ones(1,3);                                                            % permeabilities, including the host medium     
% -------------------------------------------------------------------------
% setting up output
abel = zeros( numel( lam ), numel( l ) ); 
abml = zeros( numel( lam ), numel( l ) );
abt  = zeros( numel( lam ), 1 );
exel = zeros( numel( lam ), numel( l ) ); 
exml = zeros( numel( lam ), numel( l ) );
ext  = zeros( numel( lam ), 1 );
scel = zeros( numel( lam ), numel( l ) ); 
scml = zeros( numel( lam ), numel( l ) );
sct  = zeros( numel( lam ), 1 );
% -------------------------------------------------------------------------
%% CALCULATING SCATTERING, ABSORPTION AND EXTINCTION
% -------------------------------------------------------------------------
parfor i = 1 : numel(lam)
    nb = [nau(i), nsio2, nh];                                               % using bulk n for Au
    T = t_mat( rad, nb, mu, lam(i), l );                                    % transfer matrices
    [ sc, ab, ex ] = crs_sec( rad, lam(i), nh, l, T );                      % getting sc, ab, ex
    abel(i,:) = ab.efel;  scel(i,:) = sc.efel;  exel(i,:) = ex.efel;        % writing output electric
    abml(i,:) = ab.efml;  scml(i,:) = sc.efml;  exml(i,:) = ex.efml;        % writing output magnetic
    abt(i)    = ab.efem;  sct(i)    = sc.efem;  ext(i)    = ex.efem;
end
% -------------------------------------------------------------------------
%% PLOTTING RESULTS
% -------------------------------------------------------------------------
lam = lam*1e9;
% -------------------------------------------------------------------------
figure();
% -------------------------------------------------------------------------
% electric contribution
% -------------------------------------------------------------------------
subplot(3,3,1); plot(lam,abel,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(ext,[],'all')]);
title('absorption'); ylabel('electric');
legend('$\ell=1$','$\ell=2$','$\ell=3$','Interpreter','latex');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(3,3,2); plot(lam,scel,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(ext,[],'all')]);
title('scattering');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(3,3,3); plot(lam,exel,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(ext,[],'all')]);
title('extinction');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
% magnetic contribution
% -------------------------------------------------------------------------
subplot(3,3,4); plot(lam,abml,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(ext,[],'all')]);
title('absorption'); ylabel('magnetic');
legend('$\ell=1$','$\ell=2$','$\ell=3$','Interpreter','latex');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(3,3,5); plot(lam,scml,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(ext,[],'all')]);
title('scattering');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(3,3,6); plot(lam,exml,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(ext,[],'all')]);
title('extinction');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
% total
% -------------------------------------------------------------------------
subplot(3,3,7); plot(lam,abt,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(ext,[],'all')]);
title('absorption'); ylabel('total');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(3,3,8); plot(lam,sct,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(ext,[],'all')]);
title('scattering');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(3,3,9); plot(lam,ext,'LineWidth',2);
xlim([lam(1) lam(end)]); ylim([0 max(ext,[],'all')]);
title('extinction');
xlabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
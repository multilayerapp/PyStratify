% this script calculates electromagnetic energy density inside and outside
% of the Au@SiO2@Au matryoshka
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% setting up incident illumination
lam = linspace( 400, 900, 501 )'*1e-9;                                      % wavelength
% -------------------------------------------------------------------------
% setting up particle
rad = [10,40,45]*1e-9;                                                      % outer radii of each shell
mu  = ones(1,4);                                                            % permeabilities, including the host medium
% -------------------------------------------------------------------------
% setting up refractive indices
load( 'Au_JC.mat' );                                                        % loading refractive indices
nsio2 = 1.45*ones(numel(lam),1);                                            % silica shell
nau   = complex( interp1( Au_JC(:,1), Au_JC(:,2), lam, 'spline' ),...       % interpolating tabulated data for Au
                 interp1( Au_JC(:,1), Au_JC(:,3), lam, 'spline') );          
nauc  = el_fr_pth( lam, nau, rad(1),   'Au_Ord' );                          % correction for core
naus  = el_fr_pth( lam, nau, rad(2:3), 'Au_Ord' );                          % correction for outer shell
nh    = ones(numel(lam),1);                                                 % vacuum host
nm  = [nauc,nsio2,naus,nh];                                                 % refractive indices, including the host medium
% -------------------------------------------------------------------------
% setting up properties for energy density
rw = linspace( 0, 60, 481 )*1e-9;                                           % points for calculation of energy density     
% -------------------------------------------------------------------------
% setting up output
wr_out = zeros( numel( lam ), numel( rw ) );
nrm = true;                                                                 % normalized energy
% -------------------------------------------------------------------------
%% CALCULATING ENERGY RADIAL DENSITY
% -------------------------------------------------------------------------
parfor il = 1 : numel( lam )
    l = 1 : l_conv( rad(end), nh(1), lam(il), 'near' );                     % defining l required for convergence
    T = t_mat( rad, nm(il,:), mu, lam(il), l );                             % transfer matrices
    G = G_prefac( ["Au_Ord","SiO2","Au_Ord","vac"], lam(il) );              % G prefactors
    [ wr, ~] = nrg_dns( rw, rad, nm(il,:), mu, lam(il), G, l, T, nrm );
    wr_out(il,:) = wr.e;
end
% -------------------------------------------------------------------------
%% PLOTTING RESULTS
% -------------------------------------------------------------------------
rw = rw*1e9; lam = lam*1e9;
% -------------------------------------------------------------------------
figure();
% -------------------------------------------------------------------------
imagesc(rw,lam,log10(wr_out));
colormap(hot); shading interp; colorbar;
xlim([rw(1) rw(end)]); ylim([lam(1) lam(end)]);
title('$\log[w_{n}^{(e)}(\lambda,r)/w^{(e)}_0]$','Interpreter','latex');
xlabel('$r$, nm','Interpreter','latex');
ylabel('$\lambda$, nm','Interpreter','latex');
xline(10,'-w','LineWidth',1); xline(40,'-w','LineWidth',1); xline(45,'-w','LineWidth',1);
text(4,750,'Au'); text(24,750,'SiO2'); text(42,750,'Au'); text(52,750,'air')
% -------------------------------------------------------------------------
% this script calculates total electromagnetic energy within SiO2@Au
% core-shell at different lambda and for different core radii
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% setting up incident illumination
lam = linspace(500,900,401)*1e-9;                                           % wavelength
% -------------------------------------------------------------------------
% setting up particle
rc = linspace(10,100,91)*1e-9;                                              % different core radii
ts = 5e-9;                                                                  % fixed shell thickness
mu  = ones(1,3);                                                            % permeabilities, including the host medium
% -------------------------------------------------------------------------
% setting up refractive indices
load( 'Au_JC.mat' );                                                        % loading refractive indices
nau   = complex( interp1( Au_JC(:,1), Au_JC(:,2), lam, 'spline' ),...       % interpolating tabulated data for Au
                 interp1( Au_JC(:,1), Au_JC(:,3), lam, 'spline') );          
nsio2 = 1.45;
nh    = 1;                                                                  % vacuum host
% -------------------------------------------------------------------------
w_out = zeros( numel( lam ), numel( rc ), numel( mu ) - 1 );                % pre-allocating output
nrm = true;                                                                 % normalized energy
% -------------------------------------------------------------------------
%% CALCULATING TOTAL ENERGY
% -------------------------------------------------------------------------
for il = 1 : numel(lam)
    lami = lam(il);    naui = nau(il);
    parfor ir = 1 : numel(rc)
        rad = [rc(ir),rc(ir)+ts];
        nauc = el_fr_pth( lami, naui, rad(1:2), 'Au_Ord' );                 % electron free path correction
        nm  = [nsio2,nauc,nh];                                              % shell-by-shell refractive index, including the host medium
        l = 1 : l_conv( rad(end), nh, lami, 'near' );                       % defining l required for convergence
        T = t_mat( rad, nm, mu, lami, l );                                  % transfer matrices
        G = G_prefac( ["SiO2","Au_Ord","vac"], lami );                      % G prefactors
        w = nrg_tot( rad, nm, mu, lami, G, l, T, nrm );
        w_out(il,ir,:) = w.e;
    end
end
% -------------------------------------------------------------------------
%% PLOTTING RESULTS
% -------------------------------------------------------------------------
rc = rc*1e9; lam = lam*1e9;
% -------------------------------------------------------------------------
figure();
% -------------------------------------------------------------------------
subplot(1,2,1); imagesc(rc,lam,squeeze(log10(w_out(:,:,1))));
colormap(hot); shading interp; colorbar; caxis([0 4]);
xlim([rc(1) rc(end)]); ylim([lam(1) lam(end)]);
title('Core, $\log[W_{1}^{(e)}(\lambda,r)/W^{(e)}_0]$','Interpreter','latex');
xlabel('$r_c$, nm','Interpreter','latex');
ylabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
subplot(1,2,2); imagesc(rc,lam,squeeze(log10(w_out(:,:,2))));
colormap(hot); shading interp; colorbar; caxis([0 4]);
xlim([rc(1) rc(end)]); ylim([lam(1) lam(end)]);
title('Shell, $\log[W_{2}^{(e)}(\lambda,r)/W^{(e)}_0]$','Interpreter','latex');
xlabel('$r_c$, nm','Interpreter','latex');
ylabel('$\lambda$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
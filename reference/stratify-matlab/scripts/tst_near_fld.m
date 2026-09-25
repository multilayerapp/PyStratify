% this script calculates electric and magnetic field intensities for 
% SiO2-Au matryoshkas and reproduces a part of Fig.3 from:
%
% L. Meng, R. Yu, M. Qiu, and F. J. García de Abajo, 
% "Plasmonic nano-oven by concatenation of multishell photothermal enhancement," 
% ACS Nano 11(8), 7915?7924 (2017)
% http://pubs.acs.org/doi/pdf/10.1021/acsnano.7b02426
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% setting up incident illumination
lam = 690e-9;                                                               % wavelength
% -------------------------------------------------------------------------
% setting up particles
rad = cell(3,1);                                                            % allocating cell array
rad{1} = [23,28]*1e-9;                                                      % SiO2@Au
rad{2} = [10,13,36,48]*1e-9;                                                % SiO2@Au@SiO2@Au
rad{3} = [4,7,10,16,34,42]*1e-9;                                            % SiO2@Au@SiO2@Au@SiO2@Au
% -------------------------------------------------------------------------
% setting up refractive indices
load( 'Au_JC.mat' );                                                        % loading Au Johnson and Christy
nau = complex( interp1( Au_JC(:,1), Au_JC(:,2), lam, 'spline' ),...         % interpolating data for Au
               interp1( Au_JC(:,1), Au_JC(:,3), lam, 'spline') ).';   
nh = 1.33;                                                                  % host medium
nsio2 = 1.45;                                                               % SiO2
% -------------------------------------------------------------------------
% setting up mesh for near-field
xl = linspace(-70,70,700)*1e-9;                                             % 140x140nm^2 area in XOY plane
yl = linspace(-70,70,700)*1e-9;
zl = 0;
[X,Y,Z] = meshgrid(xl,yl,zl);
% -------------------------------------------------------------------------
%% CALCULATING NEAR-FIELDS AND PLOTTING RESULTS
% -------------------------------------------------------------------------
figure()
% -------------------------------------------------------------------------
for i = 1 : 3
% -------------------------------------------------------------------------
% calculating    
    nb = [repmat([nsio2,nau],1,i),nh];                                      % refractive indices
    mu = ones(1,2*i+1);                                                     % permeabilities        
    l = 1 : l_conv( rad{i}(end), nh, lam, 'near' );                         % defining l required for convergence
    T = t_mat( rad{i}, nb, mu, lam, l );                                    % transfer-matrices
    [~,~,I] = near_fld(rad{i}, nb, mu, lam, l, T, X,Y,Z);                   % returning field intensities
% -------------------------------------------------------------------------
% plotting
    % electric
    subplot(2,3,i); imagesc(xl*1e9, yl*1e9, log10(I.E));
    for ic = 1 : 2*i
        rectangle('Position', [-rad{i}(ic), -rad{i}(ic), 2*rad{i}(ic), 2*rad{i}(ic)]*1e9, ...
                  'Curvature', [1,1], 'EdgeColor','w')
    end
    c = colorbar('northoutside'); colormap jet; caxis([0 4]);
    title(c,'$\log(|E|^2/|E_0|^2)$','Interpreter','latex');
    xlabel('$X$, nm','Interpreter','latex');
    ylabel('$Y$, nm','Interpreter','latex');
    % magnetic
    subplot(2,3,i+3); imagesc(xl*1e9, yl*1e9, log10(I.H));
    for ic = 1 : 2*i
        rectangle('Position', [-rad{i}(ic), -rad{i}(ic), 2*rad{i}(ic), 2*rad{i}(ic)]*1e9, ...
                  'Curvature', [1,1], 'EdgeColor','w')
    end
    c = colorbar('northoutside'); colormap jet; caxis([0 3]);
    title(c,'$\log(|H|^2/|H_0|^2)$','Interpreter','latex');
    xlabel('$X$, nm','Interpreter','latex');
    ylabel('$Y$, nm','Interpreter','latex');
% -------------------------------------------------------------------------
end
% -------------------------------------------------------------------------
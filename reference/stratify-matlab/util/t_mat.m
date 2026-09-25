function T = t_mat( rad, ref, mu, lam, l )
%T_MAT calculates ordered products of E & M backward & forward t-matrices 
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% rad - outer radii for each layer of the sphere
% ref - refractive index of each shell, the last element is the host medium
% mu  - permeability of each shell, the last element is host medium
% lam - vacuum wavelength
% l   - numbers of terms in expansion (array or scalar)
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% tm - forward  magnetic
% te - forward  electric
% mm - backward magnetic
% me - backward electric
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
%% ALLOCATING USEFUL QUANTITIES
% -------------------------------------------------------------------------
ks  = 2.*pi.*ref(1:end-1)./lam;                                             % wavenumber j-th
kp  = 2.*pi.*ref(2:end)./lam;                                               % wavenumber j+1-th
xs  = ks.*rad;                                                              % size parameter x = k_j   * r_j
xp  = kp.*rad;                                                              % size parameter x = k_j+1 * r_j   
eta = ref(1:end-1)./ ref(2:end);                                            % normalized refractive index n_j / n_j+1
mu  = mu(1:end-1)./mu(2:end);                                               % normalized permeability mu_j / mu_j+1
lnm = numel(l);
% -------------------------------------------------------------------------
 Sxs  = zeros( lnm, numel( xs ) );
dSxs  = zeros( lnm, numel( xs ) );
 xixs = zeros( lnm, numel( xs ) );
dxixs = zeros( lnm, numel( xs ) );
% -------------------------------------------------------------------------
 Sxp  = zeros( lnm, numel( xp ) );
dSxp  = zeros( lnm, numel( xp ) );
 xixp = zeros( lnm, numel( xp ) );
dxixp = zeros( lnm, numel( xp ) );
% -------------------------------------------------------------------------
tmb = zeros( lnm, numel( xs ), 2, 2 );
tmf = zeros( lnm, numel( xs ), 2, 2 );
teb = zeros( lnm, numel( xs ), 2, 2 );
tef = zeros( lnm, numel( xs ), 2, 2 );
% -------------------------------------------------------------------------
%% CALCULATING RICCATI-BESSEL FUNCTIONS
% -------------------------------------------------------------------------
for i = 1 : numel( xs )
     Sxs(:,i)  =  ricbesj( l, xs(i) );
    dSxs(:,i)  = dricbesj( l, xs(i) );
     xixs(:,i) =  ricbesh( l, xs(i) );
    dxixs(:,i) = dricbesh( l, xs(i) );
end
% -------------------------------------------------------------------------
for i = 1 : numel( xp )
     Sxp(:,i)  =  ricbesj( l, xp(i) );
    dSxp(:,i)  = dricbesj( l, xp(i) );
     xixp(:,i) =  ricbesh( l, xp(i) );
    dxixp(:,i) = dricbesh( l, xp(i) );
end
% -------------------------------------------------------------------------
%% CALCULATING TRANSFER-MATRICES
% -------------------------------------------------------------------------
for i = 1 : lnm
    for j = 1 : numel( xs )
% -------------------------------------------------------------------------
        tmb(i,j,:,:) = -1i*...
            [ dxixs(i,j)*Sxp(i,j) *eta(j) - xixs(i,j)*dSxp(i,j) *mu(j), ...
              dxixs(i,j)*xixp(i,j)*eta(j) - xixs(i,j)*dxixp(i,j)*mu(j); ...
             -dSxs(i,j) *Sxp(i,j) *eta(j) + Sxs(i,j) *dSxp(i,j) *mu(j), ...
             -dSxs(i,j) *xixp(i,j)*eta(j) + Sxs(i,j) *dxixp(i,j)*mu(j) ];
% -------------------------------------------------------------------------        
        tmf(i,j,:,:) = -1i*...
            [ dxixp(i,j)*Sxs(i,j) /eta(j) - xixp(i,j)*dSxs(i,j) /mu(j), ...
              dxixp(i,j)*xixs(i,j)/eta(j) - xixp(i,j)*dxixs(i,j)/mu(j); ...
             -dSxp(i,j) *Sxs(i,j) /eta(j) + Sxp(i,j) *dSxs(i,j) /mu(j), ...
             -dSxp(i,j) *xixs(i,j)/eta(j) + Sxp(i,j) *dxixs(i,j)/mu(j) ];
% -------------------------------------------------------------------------         
        teb(i,j,:,:) = -1i*...
            [ dxixs(i,j)*Sxp(i,j) *mu(j) - xixs(i,j)*dSxp(i,j) *eta(j), ...
              dxixs(i,j)*xixp(i,j)*mu(j) - xixs(i,j)*dxixp(i,j)*eta(j); ...
             -dSxs(i,j) *Sxp(i,j) *mu(j) + Sxs(i,j) *dSxp(i,j) *eta(j), ...
             -dSxs(i,j) *xixp(i,j)*mu(j) + Sxs(i,j) *dxixp(i,j)*eta(j) ];
% -------------------------------------------------------------------------
        tef(i,j,:,:) = -1i*...
            [ dxixp(i,j)*Sxs(i,j) /mu(j) - xixp(i,j)*dSxs(i,j) /eta(j), ...
              dxixp(i,j)*xixs(i,j)/mu(j) - xixp(i,j)*dxixs(i,j)/eta(j); ...
             -dSxp(i,j) *Sxs(i,j) /mu(j) + Sxp(i,j) *dSxs(i,j) /eta(j), ...
             -dSxp(i,j) *xixs(i,j)/mu(j) + Sxp(i,j) *dxixs(i,j)/eta(j) ];
% -------------------------------------------------------------------------
    end
end
% -------------------------------------------------------------------------
%% CALCULATING ORDERED PRODUCTS OF TRANSFER-MATRICES
% -------------------------------------------------------------------------
T.tm = ones( size( tmf ) );
T.te = ones( size( tef ) );
T.mm = ones( size( tmb ) );
T.me = ones( size( teb ) );
% -------------------------------------------------------------------------
for i = 1 : lnm
    for j = 1 : numel( xs )
        if j == 1
            T.tm(i,j,:,:) = tmf(i,j,:,:);
        else
            T.tm(i,j,:,:) = squeeze( tmf(i,j,:,:) )*squeeze( T.tm(i,j-1,:,:) );
        end
    end
end
% -------------------------------------------------------------------------
for i = 1 : lnm
    for j = 1 : numel( xs )
        if j == 1
            T.te(i,j,:,:) = tef(i,j,:,:);
        else
            T.te(i,j,:,:) = squeeze( tef(i,j,:,:) )*squeeze( T.te(i,j-1,:,:) );
        end
    end
end
% -------------------------------------------------------------------------
tmb = flip( tmb, 2 );
for i = 1 : lnm
    for j = 1 : numel( xs )
        if j == 1
            T.mm(i,j,:,:) = tmb(i,j,:,:);
        else
            T.mm(i,j,:,:) = squeeze( tmb(i,j,:,:) )*squeeze( T.mm(i,j-1,:,:) );
        end
    end
end
T.mm = flip( T.mm, 2 );
% -------------------------------------------------------------------------
teb = flip( teb, 2 );
for i = 1 : lnm
    for j = 1 : numel( xs )
        if j == 1
            T.me(i,j,:,:) = teb(i,j,:,:);
        else
            T.me(i,j,:,:) = squeeze( teb(i,j,:,:) )*squeeze( T.me(i,j-1,:,:) );
        end
    end
end
T.me = flip( T.me, 2 );
% -------------------------------------------------------------------------
end
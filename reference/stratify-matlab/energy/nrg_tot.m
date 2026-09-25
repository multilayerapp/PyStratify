function w_out = nrg_tot( rad, ref, mu, lam, G, l, T, nrm )
%NRG_TOT calculates electric and magnetic energy stored within each
%                                           shell of multilayered sphere
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% rad - array with radius of each shell, meters
% ref - refractive index of each shell, the last element is the host medium
% mu  - permeability of each shell, the last element is the host medium
% lam - vacuum wavelength of incident illumination, meters
% G   - structure with electric and magnetic prefactors for energy
% l   - terms for expansion
% T   - transfer matrices
% nrm - normalization true/false
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% w_out - structure with normalized electric (e), magnetic (m), and
%         electromagnetic (em) energies stored within each shell
% -------------------------------------------------------------------------
%% REFERENCE
% -------------------------------------------------------------------------
% The code is based on paper:
% Ilia L.﻿Rasskazov, Alexander Moroz, P. Scott Carney,
% "Electromagnetic energy in multilayered spherical particles",
% Journal of the Optical Society of America A 36(9) (2019)﻿1591-1601
% https://doi.org/10.1364/JOSAA.36.001591
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
%% CALCULATING USEFUL VALUES
% -------------------------------------------------------------------------
k     = 2.*pi.*ref(1:end-1)./lam;                                           % wave vector
r.u   = rad;                                                                % radii for upper boundary of integrals
r.b   = [0,rad(1:end-1)];                                                   % radii for lower boundary of integrals
x.u   = k.*r.u;                                                             % size parameters
x.b   = k.*r.b;
l_max = max(l);                                                             % cut-off
perm  = ref(1:end-1).^2./mu(1:end-1);                                       % epsilon
epsmu = abs( perm )./abs( mu(1:end-1) );                                    % |epsilon| / |mu|
eps_h = ref(end)^2/mu(end);
sm    = find( imag( ref(1:end-1) ) == 0 );
bg    = find( imag( ref(1:end-1) ) ~= 0 );
% -------------------------------------------------------------------------
%% PRE-CALCULATING OF THE BESSEL FUNCTIONS
% -------------------------------------------------------------------------
jl.u = zeros( l_max+3, numel( x.u ) );
hl.u = zeros( l_max+3, numel( x.u ) );
jl.b = zeros( l_max+3, numel( x.b ) );
hl.b = zeros( l_max+3, numel( x.b ) );
% -------------------------------------------------------------------------
for i = 1 : numel( x.u )
    jl.u(:,i) = sbesselj( 0:l_max+2, x.u(i) );
    hl.u(:,i) = sbesselh( 0:l_max+2, x.u(i) );
end
for i = 1 : numel( x.b )
    jl.b(:,i) = sbesselj( 0:l_max+2, x.b(i) );
    hl.b(:,i) = sbesselh( 0:l_max+2, x.b(i) );
end
% -------------------------------------------------------------------------
%% BUILDING EXPANSION COEFFICIENTS (preliminary, eq.21-22)
% -------------------------------------------------------------------------
A.e = zeros( l_max, numel( rad ) );
A.m = zeros( l_max, numel( rad ) );
B.e = zeros( l_max, numel( rad ) );
B.m = zeros( l_max, numel( rad ) );
% -------------------------------------------------------------------------
for i = 1 : numel( rad )     
    if i == 1 
        A.e(:,i) = 1./squeeze( T.te(:,end,1,1) );
        A.m(:,i) = 1./squeeze( T.tm(:,end,1,1) );
    else
        A.e(:,i) = squeeze( T.me(:,i,1,1) + T.me(:,i,1,2).*T.te(:,end,2,1)./...
                            T.te(:,end,1,1) );
        B.e(:,i) = squeeze( T.me(:,i,2,1) + T.me(:,i,2,2).*T.te(:,end,2,1)./...
                            T.te(:,end,1,1) );
        A.m(:,i) = squeeze( T.mm(:,i,1,1) + T.mm(:,i,1,2).*T.tm(:,end,2,1)./...
                            T.tm(:,end,1,1) );
        B.m(:,i) = squeeze( T.mm(:,i,2,1) + T.mm(:,i,2,2).*T.tm(:,end,2,1)./...
                            T.tm(:,end,1,1) );
    end
end
% -------------------------------------------------------------------------
%% BUILDING COEFFICIENTS
% -------------------------------------------------------------------------
l_2   = repmat( 2*(1:l_max)' + 1, 1, numel( rad ) );
l_1   = repmat(   (1:l_max)' + 1, 1, numel( rad ) );
l_0   = repmat(   (1:l_max)'    , 1, numel( rad ) );
% -------------------------------------------------------------------------
%% BUILDING F FOR LOSSY SHELLS (preliminary eq.31)
% -------------------------------------------------------------------------
if ( ~isempty( bg ) )                                                      % checking if there are lossy shells
% -------------------------------------------------------------------------
F = F_lossy( A, B, x, bg, jl, hl, l_max );
fr.u = 0.5*r.u(bg).^3./(x.u(bg).^2 - conj( x.u(bg) ).^2);
fr.b = 0.5*r.b(bg).^3./(x.b(bg).^2 - conj( x.b(bg) ).^2);
w.eu = fr.u.*G.e(bg)'.*sum( l_2(:,bg).*F.msamu + l_1(:,bg).*F.eminu + l_0(:,bg).*F.eplsu );
w.eb = fr.b.*G.e(bg)'.*sum( l_2(:,bg).*F.msamb + l_1(:,bg).*F.eminb + l_0(:,bg).*F.eplsb );
w.mu = fr.u.*G.m(bg)'.*sum( l_2(:,bg).*F.esamu + l_1(:,bg).*F.mminu + l_0(:,bg).*F.mplsu );
w.mb = fr.b.*G.m(bg)'.*sum( l_2(:,bg).*F.esamb + l_1(:,bg).*F.mminb + l_0(:,bg).*F.mplsb );
if bg(1) == 1                                                               % check for core
    w.eb(1) = 0;
    w.mb(1) = 0;
end
% -------------------------------------------------------------------------
%% ENERGY FOR LOSSY SHELLS
% -------------------------------------------------------------------------
w_out.e(bg) =  w.eu - w.eb;
w_out.m(bg) = (w.mu - w.mb).*epsmu(bg);
% -------------------------------------------------------------------------
end
% -------------------------------------------------------------------------
%% BUILDING F FOR LOSLESS SHELLS
% -------------------------------------------------------------------------
if ( ~isempty( sm ) )                                                      % checking if there are lossless shells
% -------------------------------------------------------------------------
F = F_lossless( A, B, x, sm, jl, hl, l_max );
% -------------------------------------------------------------------------
fr.u = 0.25*r.u(sm).^3./x.u(sm);
fr.b = 0.25*r.b(sm).^3./x.b(sm);
w.eu = fr.u.*G.e(sm)'.*sum( l_2(:,sm).*F.msamu + l_1(:,sm).*F.eminu + l_0(:,sm).*F.eplsu );
w.eb = fr.b.*G.e(sm)'.*sum( l_2(:,sm).*F.msamb + l_1(:,sm).*F.eminb + l_0(:,sm).*F.eplsb );
w.mu = fr.u.*G.m(sm)'.*sum( l_2(:,sm).*F.esamu + l_1(:,sm).*F.mminu + l_0(:,sm).*F.mplsu );
w.mb = fr.b.*G.m(sm)'.*sum( l_2(:,sm).*F.esamb + l_1(:,sm).*F.mminb + l_0(:,sm).*F.mplsb );
if sm(1) == 1                                                               % check for core
    w.eb(1) = 0;
    w.mb(1) = 0;
end
% -------------------------------------------------------------------------
%% ENERGY FOR LOSSLESS SHELLS
% -------------------------------------------------------------------------
w_out.e(sm) =  w.eu - w.eb;
w_out.m(sm) = (w.mu - w.mb).*epsmu(sm);
% -------------------------------------------------------------------------
end
% -------------------------------------------------------------------------
%% CONSTRUCTING OUTPUT & NORMALIZING TO VOLUME OF EACH SHELL IF NEEDED
% -------------------------------------------------------------------------
if nrm
    w_out.e  = 3.*w_out.e./[ rad(1).^3, diff( rad.^3 ) ]./eps_h; % to total E energy
    w_out.m  = 3.*w_out.m./[ rad(1).^3, diff( rad.^3 ) ]./eps_h; % to total M energy
    w_out.em = (w_out.e + w_out.m)./2; % 1/2 because normalizing to total EM energy
else % but still normalized to E0
    w_out.e = pi.*w_out.e;
    w_out.m = pi.*w_out.m;
    w_out.em = w_out.e + w_out.m;
end
% ------------------------------------------------------------------------- 
end
function [ wr, I ] = nrg_dns( r_w, rad, ref, mu, lam, G, l, T, nrm )
%NRG_DNS calculates normalized electric and magnetic energy density
%                               within or outside of a multilayered sphere
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% r_w - array with points for calculating fields
% rad - array with radius of each shell, meters
% ref - refractive index of each shell, the last element is host medium
% mu  - permeability of each shell, the last element is host medium
% lam - vacuum wavelength of incident illumination, meters
% G   - structure with electric and magnetic prefactors
% l   - terms for expansion
% T   - transfer matrices
% nrm - normalization true/false
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% wr - structure with normalized electric (e), magnetic (m), and
%      electromagnetic (em) energy density at given points r_w
% I  - structure with normalized intensities of electric (e) and magnetic (m)
%      fields at given points r_w
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
k     = 2.*pi.*ref./lam;
perm  = ref.^2./mu;
x     = zeros( numel( r_w ), 1 );
epsmu = zeros( 1, numel( r_w ) );
ge    = zeros( 1, numel( r_w ) );
gm    = zeros( 1, numel( r_w ) );
% -------------------------------------------------------------------------
for i = 1 : numel( rad ) + 1
    if i == 1 
        ir    = find( and( (r_w >= 0), (r_w <= rad(i)) ) );
    elseif i <= numel( rad )
        ir    = find( and( (r_w >= rad(i-1)), (r_w <= rad(i)) ) );
    else 
        ir    = find( r_w > rad(end) );
    end 
    x(ir) = k(i).*r_w(ir);
    epsmu(ir) = abs( perm(i) )/abs( mu(i) );
    ge(ir) = G.e(i);
    gm(ir) = G.m(i);
end
% -------------------------------------------------------------------------
l_max = numel(l);
% -------------------------------------------------------------------------
%% PRE-CALCULATION OF THE BESSEL FUNCTIONS
% -------------------------------------------------------------------------
jl = zeros( l_max+3, numel( x ) );
hl = zeros( l_max+3, numel( x ) );
% -------------------------------------------------------------------------
for i = 1 : numel( x )
    jl(:,i) = sbesselj( 0:l_max+2, x(i) );
    if ( x(i) == 0 )
        hl(:,i) = zeros( l_max+3, 1 );
    else
        hl(:,i) = sbesselh( 0:l_max+2, x(i) );
    end
end
% -------------------------------------------------------------------------
%% BUILDING UP EXPANSION COEFFICIENTS
% -------------------------------------------------------------------------
Ae = zeros( l_max, numel( x ) );
Am = zeros( l_max, numel( x ) );
Be = zeros( l_max, numel( x ) );
Bm = zeros( l_max, numel( x ) );
% -------------------------------------------------------------------------
for i = 1 : numel( rad ) % loop within a sphere     
    if i == 1
        ir   = find( and( (r_w >= 0), (r_w <= rad(i)) ) );
        Ae(:,ir) = 1./repmat( squeeze( T.te(:,end,1,1) ), 1, numel( ir ) );
        Am(:,ir) = 1./repmat( squeeze( T.tm(:,end,1,1) ), 1, numel( ir ) );
    else
        ir   = find( and( (r_w >= rad(i-1)), (r_w <= rad(i)) ) );  
        Ae(:,ir) = repmat( squeeze( T.me(:,i,1,1) + T.me(:,i,1,2).*T.te(:,end,2,1)./...
                            T.te(:,end,1,1) ), 1, numel( ir ) );
        Be(:,ir) = repmat( squeeze( T.me(:,i,2,1) + T.me(:,i,2,2).*T.te(:,end,2,1)./...
                            T.te(:,end,1,1) ), 1, numel( ir ) );
        Am(:,ir) = repmat( squeeze( T.mm(:,i,1,1) + T.mm(:,i,1,2).*T.tm(:,end,2,1)./...
                            T.tm(:,end,1,1) ), 1, numel( ir ) );
        Bm(:,ir) = repmat( squeeze( T.mm(:,i,2,1) + T.mm(:,i,2,2).*T.tm(:,end,2,1)./...
                            T.tm(:,end,1,1) ), 1, numel( ir ) );
    end
end
% -------------------------------------------------------------------------
if r_w(end) > rad(end) % if outside a sphere
    ir   = find( r_w > rad(end) );
    Ae(:,ir) = ones( l_max, numel( ir ) );
    Be(:,ir) = repmat( squeeze( T.te(:,end,2,1)./T.te(:,end,1,1) ), 1, numel( ir ) );
    Am(:,ir) = ones( l_max, numel( ir ) );
    Bm(:,ir) = repmat( squeeze( T.tm(:,end,2,1)./T.tm(:,end,1,1) ), 1, numel( ir ) );
end
% -------------------------------------------------------------------------
%% BUILDING COEFFICIENTS
% -------------------------------------------------------------------------
l_2   = repmat( 2*(1:l_max)' + 1, 1, numel( x ) );
l_1   = repmat(   (1:l_max)' + 1, 1, numel( x ) );
l_0   = repmat(   (1:l_max)'    , 1, numel( x ) );
% -------------------------------------------------------------------------
%% BUILDING f
% -------------------------------------------------------------------------
f.emin = abs( Ae.*jl(1:end-3,:) + Be.*hl(1:end-3,:) ).^2;
f.mmin = abs( Am.*jl(1:end-3,:) + Bm.*hl(1:end-3,:) ).^2;
% -------------------------------------------------------------------------
f.esam = abs( Ae.*jl(2:end-2,:) + Be.*hl(2:end-2,:) ).^2;
f.msam = abs( Am.*jl(2:end-2,:) + Bm.*hl(2:end-2,:) ).^2;
% -------------------------------------------------------------------------
f.epls = abs( Ae.*jl(3:end-1,:) + Be.*hl(3:end-1,:) ).^2;
f.mpls = abs( Am.*jl(3:end-1,:) + Bm.*hl(3:end-1,:) ).^2;
% -------------------------------------------------------------------------
%% CONSTRUCTING OUTPUT & NORMALIZING
% -------------------------------------------------------------------------
I.e  = sum( l_2.*f.msam + l_1.*f.emin + l_0.*f.epls )/2;
I.m  = sum( l_2.*f.esam + l_1.*f.mmin + l_0.*f.mpls )/2;
% -------------------------------------------------------------------------
wr.e =        pi.*ge.*I.e; % but still normalized to E0
wr.m = epsmu.*pi.*gm.*I.m;
wr.em = (wr.e + wr.m);
% -------------------------------------------------------------------------
if nrm
    wr.e = wr.e/perm(end)/pi; % to E density
    wr.m = wr.m/perm(end)/pi; % to M density
    wr.em = (wr.e + wr.m)./2; % to EM density
end
% -------------------------------------------------------------------------
end
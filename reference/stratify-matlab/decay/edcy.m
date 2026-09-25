function [ gr, gnr ] = edcy( rad, dip, ref, mu, lam, l, rin, T, nrm )
%DECAY calculates radiative and nonradiative decay rates of the dipole
% located inside or outside of a multilayered sphere
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% rad - outer radii for each layer of the sphere
% dip - positions of the dipole
% ref - refractive indices for each layer + refractive index of
%       the surrounding medium (last element of n)
% mu  - magnetic permeabilities for each layer and magnetic permeability of
%       the surrounding medium (last element of mu)
% lam - vacuum wavelength
% l   - numbers of terms in exansion (array or scalar)
% rin - number of integration points for integrals in I_abs
% T   - transfer matrices, see 't_mat.m'
% tol - tolerance, convergence criteria for decay rates
% nrm - either 'host' or 'shell', normalization of decay rates to the host
%       medium or to the shell where the dipole is located
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% gr  - radiative    decay rates: 1 - normal, 2 - parallel
% gnr - nonradiative decay rates: 1 - normal, 2 - parallel
% -------------------------------------------------------------------------
%% REFERENCE
% -------------------------------------------------------------------------
% The code is based on paper:
% Alexander Moroz, "A recursive transfer-matrix solution for a dipole 
% radiating inside and outside a stratified sphere",
% Annals of Physics 315 (2005) 352–418
% http://linkinghub.elsevier.com/retrieve/pii/S0003491604001277
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
%% DEFINE POSITIONS OF THE DIPOLE
% -------------------------------------------------------------------------
x = zeros( 1, numel(dip) );
d = zeros( 1, numel(dip) );
k = zeros( 1, numel(ref) );
% -------------------------------------------------------------------------
for i = 1 : numel( rad ) % within a sphere     
    if i == 1 
        rmin = 0;
        rmax = rad(i);
    else
        rmin = rad(i-1);
        rmax = rad(i);
    end 
    ir    = find( and( (dip >= rmin), (dip < rmax) ) );
    k(i) = 2*pi*ref(i)./lam;
    x(ir) = k(i).*dip(ir);
    d(ir) = i;
end
% -------------------------------------------------------------------------
ir = find( dip >= rad(end) ); % outside of a sphere
k(end) = 2*pi*ref(end)./lam;
x(ir) = k(end).*dip(ir);
d(ir) = numel(ref);
% -------------------------------------------------------------------------
%% SET NORMALIZATION COEFFICIENTS FOR DECAY RATES
% -------------------------------------------------------------------------
dun = unique(d);
rcf  = zeros( numel( dun ), 1 );
nrcf = zeros( numel( dun ), 1 );
for i = 1 : numel( dun )
    di = dun(i);
    switch nrm
    case 'host' 
        rcf(i)  = ref(di)*mu(di)/ref(end)/mu(end);
        nrcf(i) = mu(di)^2/(ref(end)*mu(end)*ref(di));
    case 'shell'
        rcf(i)  = (ref(di)*mu(di)/ref(end)/mu(end))^2;
        nrcf(i) = mu(di)/(ref(di)^2);
    otherwise
        error('Unexpected normalization. Exiting.')
    end
end
% -------------------------------------------------------------------------
%% CALCULATING USEFUL QUANTITIES
% -------------------------------------------------------------------------
cffl = ( l.*(l+1).*(2.*l+1) )'; % coefficients
cfrd = ( 2.*l+1 )';
nabs = find(imag(ref) > 0);    % finding absorbing layers
grcumtmp  = zeros( numel( l ), 2 );
gnrcumtmp = zeros( numel( l ), numel(nabs), 2 );
gr   = zeros( numel( dip ), 2 ); % allocating arrays for output
gnr  = zeros( numel( dip ), 2 );
% -------------------------------------------------------------------------
[ Im, Ie ] = I_abs( rad, d, ref, lam, l, T, rin ); % integrals
% -------------------------------------------------------------------------
%% LOOP OVER DIPOLE POSITIONS
% -------------------------------------------------------------------------
for i = 1 : numel( dip )
% -------------------------------------------------------------------------
%   common variables
% -------------------------------------------------------------------------
    di = d(i);
% -------------------------------------------------------------------------
     Sx  =  ricbesj( l, x(i) ).';
    dSx  = dricbesj( l, x(i) ).';
     xix =  ricbesh( l, x(i) ).';
    dxix = dricbesh( l, x(i) ).';
% -------------------------------------------------------------------------
%   radiative decay rates
% -------------------------------------------------------------------------   
    if di == numel(ref) % outside the sphere
         fel =  Sx + (T.te(:,end,2,1)./T.te(:,end,1,1)).* xix;
        dfel = dSx + (T.te(:,end,2,1)./T.te(:,end,1,1)).*dxix;
         fml =  Sx + (T.tm(:,end,2,1)./T.tm(:,end,1,1)).* xix;        
    elseif di == 1 % inside the core
         fel =  Sx./T.me(:,1,2,2);
        dfel = dSx./T.me(:,1,2,2);
         fml =  Sx./T.mm(:,1,2,2);        
    else % within a shell, except core
        fel = (T.te(:,di-1,1,1).*Sx + T.te(:,di-1,2,1).*xix)./...
            (T.me(:,di,2,2).*T.te(:,di-1,1,1) - T.me(:,di,1,2).*T.te(:,di-1,2,1));
        dfel = (T.te(:,di-1,1,1).*dSx + T.te(:,di-1,2,1).*dxix)./...
            (T.me(:,di,2,2).*T.te(:,di-1,1,1) - T.me(:,di,1,2).*T.te(:,di-1,2,1));
        fml = (T.tm(:,di-1,1,1).*Sx + T.tm(:,di-1,2,1).*xix)./...
            (T.mm(:,di,2,2).*T.tm(:,di-1,1,1) - T.mm(:,di,1,2).*T.tm(:,di-1,2,1));
    end
% -------------------------------------------------------------------------
%   nonradiative decay rates
% -------------------------------------------------------------------------
for j = 1 : numel( nabs )
    if di == numel(ref) % outside the sphere
        del  =  xix;
        ddel = dxix;
        dml  =  xix;
    elseif di == 1 % inside the sphere core
        del  =  Sx./T.me(:,1,2,2);
        ddel = dSx./T.me(:,1,2,2);
        dml  =  Sx./T.mm(:,1,2,2);
    elseif di > nabs(j) % within a shell, except core
        del = (T.me(:,di,1,2).*Sx + T.me(:,di,2,2).*xix)./...
            (T.me(:,di,2,2).*T.te(:,di-1,1,1) - T.me(:,di,1,2).*T.te(:,di-1,2,1));
        ddel = (T.me(:,di,1,2).*dSx + T.me(:,di,2,2).*dxix)./...
            (T.me(:,di,2,2).*T.te(:,di-1,1,1) - T.me(:,di,1,2).*T.te(:,di-1,2,1));
        dml = (T.mm(:,di,1,2).*Sx + T.mm(:,di,2,2).*xix)./...
            (T.mm(:,di,2,2).*T.tm(:,di-1,1,1) - T.mm(:,di,1,2).*T.tm(:,di-1,2,1));
    else 
        del = (T.te(:,di-1,1,1).*Sx + T.te(:,di-1,2,1).*xix)./...
            (T.me(:,di,2,2).*T.te(:,di-1,1,1) - T.me(:,di,1,2).*T.te(:,di-1,2,1));
        ddel = (T.te(:,di-1,1,1).*dSx + T.te(:,di-1,2,1).*dxix)./...
            (T.me(:,di,2,2).*T.te(:,di-1,1,1) - T.me(:,di,1,2).*T.te(:,di-1,2,1));
        dml = (T.tm(:,di-1,1,1).*Sx + T.tm(:,di-1,2,1).*xix)./...
            (T.mm(:,di,2,2).*T.tm(:,di-1,1,1) - T.mm(:,di,1,2).*T.tm(:,di-1,2,1));
    end
% -------------------------------------------------------------------------    
    gnrc(1) = (3*k(di)^3)/(2*x(i)^4)*nrcf( dun == di);
    gnrc(2) = (3*k(di)^3)/(4*x(i)^2)*nrcf( dun == di);
    dnum = find( unique(d) == di );
    gnrcumtmp(:,j,1) = gnrc(1)*cumsum(cffl.*Ie(:,j,dnum).*abs(del).^2);     % perpendicular
    gnrcumtmp(:,j,2) = gnrc(2)*cumsum(cfrd.*(Ie(:,j,dnum).*abs(ddel).^2 + Im(:,j,dnum).*abs(dml).^2));   % parallel
end
% -------------------------------------------------------------------------
%   constructing output
% -------------------------------------------------------------------------
    grc(1) = 3/(2*x(i)^4)*rcf( dun == di);
    grc(2) = 3/(4*x(i)^2)*rcf( dun == di);
    grcumtmp(:,1) = grc(1)*cumsum(cffl.*abs(fel).^2);
    grcumtmp(:,2) = grc(2)*cumsum(cfrd.*(abs(dfel).^2 + abs(fml).^2));
    grcum = rmmissing(grcumtmp);
    gr(i,:)  = grcum(end,:);
% -------------------------------------------------------------------------
    gnrcum = squeeze(sum(gnrcumtmp,2)); % summarizing over absorbing shells
    gnrcum = rmmissing(gnrcum);
% -------------------------------------------------------------------------
    if isempty(grcum) || isempty(gnrcum)
        error('Error with decay rate convergence. Exiting')
    end
% -------------------------------------------------------------------------    
    if max (imag(ref),[],'all') > 1
        gnr1 = gnrcum(diff(gnrcum(:,1),1,1)./gnrcum(2:end,1)<1e-2,1);
        gnr2 = gnrcum(diff(gnrcum(:,2),1,1)./gnrcum(2:end,2)<1e-2,2);
        gnr(i,:) = [gnr1(ceil(end/10)) gnr2(ceil(end/10))];
    else
        gnr(i,:) = gnrcum(end,:);
    end
% -------------------------------------------------------------------------       
end
% -------------------------------------------------------------------------    
end
function [ Rs, Rp, Ts, Tp ] = pln_abcd( n, lam, d, theta0 )
%% abcd calculates reflection and transmission coefficients 
%                                              for multilayered structure
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% n      - matrix for complex refractive indices of layers
%          rows correspond to layers
%          columns correspond to wavenumbers
% lam    - wavelength, scalar, nanometers
% d      - thicknesses of each layer, nanometers
% theta0 - angle of incidence
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% rs     - reflection coefficent for s-polarized (TE) mode
% rp     - reflection coefficent for p-polarized (TM) mode
% ts     - transmission coefficent for s-polarized (TE) mode
% tp     - transmission coefficent for p-polarized (TM) mode
% -------------------------------------------------------------------------
%% COPYRIGHT
% -------------------------------------------------------------------------
% Copyright 2021 Ilia Rasskazov, University of Rochester
% -------------------------------------------------------------------------
% Author:        Ilia Rasskazov, irasskaz@ur.rochester.edu
% -------------------------------------------------------------------------
% Organization:  The Institute of Optics, University of Rochester
%                http://www.hajim.rochester.edu/optics/
% -------------------------------------------------------------------------
%% ALLOCATING USEFUL QUANTITIES
% -------------------------------------------------------------------------
rs = zeros(numel(d)+1,numel(lam));
rp = zeros(numel(d)+1,numel(lam));
ts = zeros(numel(d)+1,numel(lam));
tp = zeros(numel(d)+1,numel(lam));
Rs = zeros(numel(lam),1);
Rp = zeros(numel(lam),1);
Ts = zeros(numel(lam),1);
Tp = zeros(numel(lam),1);
ss = cell(numel(d)+1,numel(lam));
sp = cell(numel(d)+1,numel(lam));
Ss = cell(1,numel(lam));
Sp = cell(1,numel(lam));      
% -------------------------------------------------------------------------
%% CALCULATING PHASE SHIFTS FOR EACH LAYER
% -------------------------------------------------------------------------
theta_full = zeros( size( n, 1 ) - 1, size( n, 2 ) );                      
theta_full(1,:) = theta0;
% -------------------------------------------------------------------------
for i = 2:size( n, 1 ) - 1
    theta_full(i,:) = asin( n(i-1,:)./n(i,:).*sin( theta_full(i-1,:) ) );   % Snell's law
end
% -------------------------------------------------------------------------
delt = 2.*pi.*n(2:end-1,:).*...
       bsxfun( @times, bsxfun( @times, d, cos( theta_full(2:end,:) ) ), 1./lam );    
edp = exp(  1i.*delt );
edm = 1./edp;
% -------------------------------------------------------------------------
%% CALCULATING REFLECTION AND TRANSMISSION COEFFICIENTS FOR EACH BOUNDARY
% -------------------------------------------------------------------------
for i = 1:numel(d)+1
    [ rs(i,:), rp(i,:), ts(i,:), tp(i,:) ] = ...
                  pln_fres( n(i,:), n(i+1,:), theta_full(i,:) );
end
% -------------------------------------------------------------------------
%% CALCULATING TRANSFER MATRIX ELEMENTS FOR EACH LAYER
% -------------------------------------------------------------------------
for i = 1:numel(lam)
    ss{1,i} = [ 1, rs(1,i); rs(1,i), 1 ] ./ ts(1,i);
    sp{1,i} = [ 1, rp(1,i); rp(1,i), 1 ] ./ tp(1,i);
    for j = 2 : numel(d) + 1
        ss{j,i} = [ edm(j-1,i), rs(j,i)*edm(j-1,i); ...
                    rs(j,i)*edp(j-1,i), edp(j-1,i) ] ./ ts(j,i);
        sp{j,i} = [ edm(j-1,i), rp(j,i)*edm(j-1,i); ...
                    rp(j,i)*edp(j-1,i), edp(j-1,i) ] ./ tp(j,i);
    end
end
% -------------------------------------------------------------------------
%% CALCULATING TRANSFER MATRIX
% -------------------------------------------------------------------------
for i = 1:numel(lam)
    Ss{1,i} = ss{1,i};
    Sp{1,i} = sp{1,i};
    for j = 2:numel(d)+1
        Ss{1,i} = Ss{1,i}*ss{j,i};
        Sp{1,i} = Sp{1,i}*sp{j,i};
    end
    Rs(i) = Ss{1,i}(2,1)/Ss{1,i}(1,1);
    Rp(i) = Sp{1,i}(2,1)/Sp{1,i}(1,1);
    Ts(i) = 1/Ss{1,i}(1,1);
    Tp(i) = 1/Sp{1,i}(1,1);
end
% -------------------------------------------------------------------------
end
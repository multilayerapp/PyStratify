function [ Im, Ie ] = I_abs( rad, d, ref, lam, l, T, rin )
%I_abs calculates integrals entering nonradiative decay rates of a dipole
% located inside or outside of a multilayered sphere
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% rad - outer radii for each layer of the sphere
% d   - array containing numbers of shells where dipoles are located
% ref - refractive indices for each layer and refractive index of
%       the surrounding medium (last element)
% lam - vacuum wavelength
% l   - numbers of terms in exansion (array or scalar)
% T   - transfer matrices, see 't_mat.m'
% rin - number of integration points for integrals in Im or Ie
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% Im, Ie - integrals from Eq.(116) entering Eqs.(132),(133) 
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
%% FINDING ABSORBING SHELLS
% -------------------------------------------------------------------------
nabs = find(imag(ref) > 0);
% -------------------------------------------------------------------------
if any(nabs == numel(ref))
    error('Surrounding medium is absorbing. Exiting');
end
% -------------------------------------------------------------------------
d = unique(d);
% -------------------------------------------------------------------------
%% ALLOCATING USEFUL QUANTITIES
% -------------------------------------------------------------------------
k = 2*pi*ref(nabs)/lam;
coef = (l.*(l+1))';
% -------------------------------------------------------------------------
Ie   = zeros( numel(l), numel(nabs), numel(d));
Im   = zeros( numel(l), numel(nabs), numel(d) );
 Sx  = zeros( numel(l), rin );
dSx  = zeros( numel(l), rin );
 xix = zeros( numel(l), rin );
dxix = zeros( numel(l), rin );
% -------------------------------------------------------------------------
%% LOOP OVER LOCATIONS OF THE DIPOLE EMITTER
% -------------------------------------------------------------------------
for dj = 1 : numel( d )
    for aj = 1 : numel( nabs )

        if d(dj) == nabs(aj)
        error('Dipole is located in absorbing shell. Exiting');
        end

        na = nabs(aj);

        if na ~= 1
            r = linspace(rad(na-1),rad(na),rin);
        else
            r = linspace(0,rad(na),rin);
        end

        ka = k(aj);
        xa = ka.*r;

        for i = 1 : rin        
            Sx(:,i)   =  sbesselj(l, xa(i));
            dSx(:,i)  = dsbesselj(l, xa(i));
            xix(:,i)  =  besselh(l, xa(i));
            dxix(:,i) = dbesselh(l, xa(i));
        end

        if na == 1
            xix(:,1) = 0;
            dxix(:,1) = 0;
            dSx(:,1) = 0;
        end

        if d(dj) == numel(ref)
            am = T.mm(:,na,1,1) + T.mm(:,na,1,2).*T.tm(:,end,2,1)./T.tm(:,end,1,1);
            ae = T.me(:,na,1,1) + T.me(:,na,1,2).*T.te(:,end,2,1)./T.te(:,end,1,1);
            bm = T.mm(:,na,2,1) + T.mm(:,na,2,2).*T.tm(:,end,2,1)./T.tm(:,end,1,1);
            be = T.me(:,na,2,1) + T.me(:,na,2,2).*T.te(:,end,2,1)./T.te(:,end,1,1);
        end

        if d(dj) < na
            am = T.mm(:,na,1,2);
            ae = T.me(:,na,1,2);
            bm = T.mm(:,na,2,2);
            be = T.me(:,na,2,2);
        end

        if d(dj) > na && d(dj) < numel(ref)
            if na == 1
                am = 1;
                ae = 1;
            else
                am = T.tm(:,na-1,1,1);
                ae = T.te(:,na-1,1,1);
                bm = T.tm(:,na-1,2,1);
                be = T.te(:,na-1,2,1);
            end
        end

        if na == 1
            bm = 0;
            be = 0;
        end

        am = repmat(am,1,rin);
        ae = repmat(ae,1,rin);
        bm = repmat(bm,1,rin);
        be = repmat(be,1,rin);

        r_int = repmat(r,numel(l),1);
        z_im  = abs((am.*Sx + bm.*xix)).^2.*(r_int.^2);
        z_ie1 = abs((ae.*Sx + be.*xix)).^2;
        z_ie2 = abs(ae.*(Sx + xa.*dSx) + be.*(xix + xa.*dxix)).^2;
        
        Im(:,aj,dj) = imag(ref(na)^2).*trapz(r,z_im,2);
        Ie(:,aj,dj) = imag(ref(na)^2).*(coef.*trapz(r,z_ie1,2) + trapz(r,z_ie2,2))./abs(ka)^2;

    end
end
% -------------------------------------------------------------------------
end
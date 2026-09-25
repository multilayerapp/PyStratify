function [E,H,I] = near_fld(rad, ref, mu, lam, l, T, X,Y,Z, nworkers)
%NEAR_FLD calculates electric and magnetic near-fields;
%by default, the incident wavevector k is along Z axis, and E0 is along X
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% rad - outer radii for each layer of the sphere
% ref - refractive index of each shell, the last element is the host medium
% mu  - permeability of each shell, the last element is host medium
% lam - vacuum wavelength
% l   - numbers of terms in expansion (array or scalar)
% T   - transfer matrix
% X,Y,Z - mesh coordinates
% nworkers - number of parallel workers to speed up near field calculations
%            if the Parallel Computing Toolbox is available (defaults to 0)
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% E - normalized electric field components
% H - normalized magnetic field components
%       both are returned in:
%           - Cartesian coordinates: .x, .y, .z
%           - Spherical coordinates: .r, .ph, .th
% I - normalized intensities of electric (.E) and magnetic (.H) fields
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
% from Cartesian to Spherical coordinates, not the same as cart2sph
th = pi/2 - atan(Z./sqrt(X.^2 + Y.^2));
ph = atan2(Y,X);
r = sqrt(X.^2 + Y.^2 + Z.^2);
% -------------------------------------------------------------------------
cth  = cos(th);                                                             % cosine theta for Pl
[cthu, ~, ctui] = unique( cth );                                            % finding unique cosines
sthu = sqrt(1 - cthu.^2);                                                   % sinus theta for Y's
lpl   = [l,l(end)+1];                                                       % plus extra multipole 
Plplu = zeros( numel(lpl), numel(cthu) );                                   % Legendre polynomials
% -------------------------------------------------------------------------
%% CALCULATING ASSOCIATED LEGENDRE POLYNOMIALS AND THEIR DERIVATIVES
% -------------------------------------------------------------------------
for il = 1 : numel(lpl)
    tmpPl = legendre( lpl(il), cthu ); Plplu(il,:) = tmpPl(2,:);
end
% -------------------------------------------------------------------------
Plplu = Plplu./repmat(sthu',numel(l)+1,1); % Legendre over sinus theta
if (~all(sthu))                            % resolving 0/0
    Plplu(:,sthu == 0) = -repmat((lpl.*(lpl+1)./2)',1,nnz(~sthu));
end
% -------------------------------------------------------------------------
dPlplu = repmat(l',1,numel(cthu)).*Plplu(2:end,:) ...                       % recurrence relation
       - repmat((l+1)',1,numel(cthu)).*repmat(cthu',numel(l),1).*Plplu(1:end-1,:);
% -------------------------------------------------------------------------
Pconv = - 1./(lpl.*(lpl+1));
Plmnu = repmat(Pconv',1,numel(cthu)).*Plplu;
% -------------------------------------------------------------------------
dPlmnu = repmat((l+2)',1,numel(cthu)).*Plmnu(2:end,:) ...
       - repmat((l+1)',1,numel(cthu)).*repmat(cthu',numel(l),1).*Plmnu(1:end-1,:);
% -------------------------------------------------------------------------
Plplu = Plplu(1:end-1,:); Plmnu = Plmnu(1:end-1,:);
% -------------------------------------------------------------------------
%% CONSTRUCTING Y(o) and Y(m)
% -------------------------------------------------------------------------
memn = 1i.*sqrt((2*l+1)./(4*pi));
mepl = memn./(l.*(l+1));
% -------------------------------------------------------------------------
opl  = mepl.*sqrt(l.*(l+1));
omn  = opl.*(l.*(l+1));
% -------------------------------------------------------------------------
Ympl.ph = -    repmat(mepl.',1,numel(cthu)).*dPlplu;
Ympl.th =  1i.*repmat(mepl.',1,numel(cthu)).*Plplu;
Ymmn.ph = -    repmat(memn.',1,numel(cthu)).*dPlmnu;
Ymmn.th = -1i.*repmat(memn.',1,numel(cthu)).*Plmnu;
% -------------------------------------------------------------------------
Yopl.r  = repmat(opl.',1,numel(cthu)).*Plplu.*repmat(sthu',numel(l),1);
Yomn.r  = repmat(omn.',1,numel(cthu)).*Plmnu.*repmat(sthu',numel(l),1);
% -------------------------------------------------------------------------
%% CONSTRUCTING BESSEL FUNCTIONS
% -------------------------------------------------------------------------
kr = zeros( 1, numel(r) ); nmu = zeros( size(r) );
% -------------------------------------------------------------------------
for i = 1 : numel( rad ) + 1
    if i == 1 
        ir = find( and( (r >= 0), (r < rad(i)) ) );
    elseif i <= numel( rad ) 
        ir = find( and( (r >= rad(i-1)), (r < rad(i)) ) );
    else
        ir = find( r >= rad(end) );
    end
    kr(ir) = 2*pi*ref(i)./lam.*r(ir);
    nmu(ir) = ref(i)/mu(i);
end
% -------------------------------------------------------------------------
[kru,ru_i,kru_i] = unique( kr );                                            % finding unique cosines
ru = r(ru_i);
twl = repmat(2*l'+1,1,numel(kru));
% -------------------------------------------------------------------------
jlu  = zeros( numel( l )+2, numel( kru ) );                                 % spherical Bessel functions jn for vacuum
hlu  = zeros( numel( l )+2, numel( kru ) );                                 % spherical Hankel functions hn for vacuum
djlu = zeros( numel( l ), numel( kru ) );
dhlu = zeros( numel( l ), numel( kru ) );
lkr = [0,l,l(end)+1];
% -------------------------------------------------------------------------
if ~exist('nworkers','var'), nworkers = 0; end % default nworkers to zero
r1 = rad(1);
parfor (i = 1 : numel( kru ), nworkers)
    jlu(:,i)  =  sbesselj ( lkr, kru(i) );
    djlu(:,i) = dsbesselj ( l, kru(i) );
    if ru(i) > r1
        hlu(:,i) = sbesselh ( lkr, kru(i) );
        dhlu(:,i) = dsbesselh ( l, kru(i) );
    end
end
jlkru = (jlu(1:end-2,:) + jlu(3:end,:))./twl;
hlkru = (hlu(1:end-2,:) + hlu(3:end,:))./twl;
djlkru = jlkru + djlu; dhlkru = hlkru + dhlu;
jlu = jlu(2:end-1,:);  hlu = hlu(2:end-1,:);
% -------------------------------------------------------------------------
%% EXPANDING MATRICES
% -------------------------------------------------------------------------
Cm = 1i.^l.*sqrt((2*l+1).*pi); Ce = Cm;
% -------------------------------------------------------------------------
Ae = zeros(numel(l),size(T.te,2)+1); Am = zeros(numel(l),size(T.te,2)+1);
Be = zeros(numel(l),size(T.te,2)+1); Bm = zeros(numel(l),size(T.te,2)+1);
% -------------------------------------------------------------------------
Ae(:,end) = Ce; Am(:,end) = Cm;
% -------------------------------------------------------------------------
Be(:,end) = squeeze(T.te(:,end,2,1))./squeeze(T.te(:,end,1,1)).*Ce.';
Bm(:,end) = squeeze(T.tm(:,end,2,1))./squeeze(T.tm(:,end,1,1)).*Cm.';
% -------------------------------------------------------------------------
for i = 1 : numel(rad)
    Ae(:,i) = squeeze(T.me(:,i,1,1)).*Ae(:,end) + squeeze(T.me(:,i,1,2)).*Be(:,end);    
    Am(:,i) = squeeze(T.mm(:,i,1,1)).*Am(:,end) + squeeze(T.mm(:,i,1,2)).*Bm(:,end);
    if i ~= 1
        Be(:,i) = squeeze(T.me(:,i,2,1)).*Ae(:,end) + squeeze(T.me(:,i,2,2)).*Be(:,end);
        Bm(:,i) = squeeze(T.mm(:,i,2,1)).*Am(:,end) + squeeze(T.mm(:,i,2,2)).*Bm(:,end);
    end
end
% -------------------------------------------------------------------------
Ampl = zeros(numel(l),numel(r)); Bmpl = zeros(numel(l),numel(r));
Aepl = zeros(numel(l),numel(r)); Bepl = zeros(numel(l),numel(r));
% -------------------------------------------------------------------------
for i = 1 : numel( rad ) + 1
    if i == 1 
        ir    = find( and( (r >= 0), (r < rad(i)) ) );
    elseif i <= numel( rad ) 
        ir    = find( and( (r >= rad(i-1)), (r < rad(i)) ) );
    else
        ir = find( r >= rad(end) );
    end     
    Ampl(:,ir) = repmat(Am(:,i),1,numel(ir));
    Bmpl(:,ir) = repmat(Bm(:,i),1,numel(ir));
    Aepl(:,ir) = repmat(Ae(:,i),1,numel(ir));
    Bepl(:,ir) = repmat(Be(:,i),1,numel(ir));
end
% -------------------------------------------------------------------------
Ampl = reshape( Ampl, [size(Ampl,1), size(X)] );
Bmpl = reshape( Bmpl, [size(Bmpl,1), size(X)] );
Aepl = reshape( Aepl, [size(Aepl,1), size(X)] );
Bepl = reshape( Bepl, [size(Bepl,1), size(X)] );
% -------------------------------------------------------------------------
jl    = reshape( jlu(:,kru_i),    [size(jlu,1),    size(X)] );
hl    = reshape( hlu(:,kru_i),    [size(hlu,1),    size(X)] );
jlkr  = reshape( jlkru(:,kru_i),  [size(jlkru,1),  size(X)] );
hlkr  = reshape( hlkru(:,kru_i),  [size(hlkru,1),  size(X)] );
djlkr = reshape( djlkru(:,kru_i), [size(djlkru,1), size(X)] );
dhlkr = reshape( dhlkru(:,kru_i), [size(dhlkru,1), size(X)] );
% -------------------------------------------------------------------------
exppl = repmat(shiftdim(exp(1i.*ph),-1),numel(l),1,1,1);
Ympl.ph = reshape( Ympl.ph(:,ctui), [size(Ympl.ph,1),size(ph)] ).*exppl;
Ympl.th = reshape( Ympl.th(:,ctui), [size(Ympl.th,1),size(th)] ).*exppl;
Ymmn.ph = reshape( Ymmn.ph(:,ctui), [size(Ymmn.ph,1),size(ph)] )./exppl;
Ymmn.th = reshape( Ymmn.th(:,ctui), [size(Ymmn.th,1),size(th)] )./exppl;
Yopl.r  = reshape( Yopl.r(:,ctui),  [size(Yopl.r,1),size(r)] ).*exppl;
Yomn.r  = reshape( Yomn.r(:,ctui),  [size(Yomn.r,1),size(r)] )./exppl;
% -------------------------------------------------------------------------
Yepl.ph = Ympl.th; Yepl.th = -Ympl.ph; Yemn.ph = Ymmn.th; Yemn.th = -Ymmn.ph;
% -------------------------------------------------------------------------
%% CONSTRUCTING OUTPUT
% -------------------------------------------------------------------------
prel = repmat(sqrt(l.*(l+1))',[1,size(r)]);
% -------------------------------------------------------------------------
% Note: Ampl = Ammn; % Bmpl = Bmmn % Aepl = -Aemn % Bepl = -Bemn
Ee.ph =       (Aepl.*djlkr + Bepl.*dhlkr).*(Yepl.ph - Yemn.ph);
Ee.th =       (Aepl.*djlkr + Bepl.*dhlkr).*(Yepl.th - Yemn.th);
Ee.r  = prel.*(Aepl.* jlkr + Bepl.* hlkr).*(Yopl.r  - Yomn.r);
Em.ph =       (Ampl.* jl   + Bmpl.* hl)  .*(Ympl.ph + Ymmn.ph);
Em.th =       (Ampl.* jl   + Bmpl.* hl)  .*(Ympl.th + Ymmn.th);
% -------------------------------------------------------------------------
E.ph = reshape(sum(Ee.ph + Em.ph,1),size(ph));
E.th = reshape(sum(Ee.th + Em.th,1),size(th));
E.r  = reshape(sum(Ee.r,1),size(r));
% -------------------------------------------------------------------------
He.ph =       (Aepl.* jl     + Bepl.* hl)  .*(Ympl.ph - Ymmn.ph);
He.th =       (Aepl.* jl     + Bepl.* hl)  .*(Ympl.th - Ymmn.th);
Hm.r  = prel.*(Ampl.* jlkr   + Bmpl.* hlkr).*(Yopl.r  + Yomn.r);
Hm.ph =       (Ampl.*djlkr   + Bmpl.*dhlkr).*(Yepl.ph + Yemn.ph);
Hm.th =       (Ampl.*djlkr   + Bmpl.*dhlkr).*(Yepl.th + Yemn.th);
% -------------------------------------------------------------------------
H.ph = -1i.*nmu.*reshape(sum(He.ph + Hm.ph,1),size(ph));
H.th = -1i.*nmu.*reshape(sum(He.th + Hm.th,1),size(th));
H.r  = -1i.*nmu.*reshape(sum(Hm.r,1),size(r));
% -------------------------------------------------------------------------
% converting to Cartesian
% -------------------------------------------------------------------------
E.x = -sin(ph).*E.ph + cos(th).*cos(ph).*E.th + sin(th).*cos(ph).*E.r;
E.y =  cos(ph).*E.ph + cos(th).*sin(ph).*E.th + sin(th).*sin(ph).*E.r;
E.z =                - sin(th).*         E.th + cos(th).*         E.r;
% -------------------------------------------------------------------------
H.x = -sin(ph).*H.ph + cos(th).*cos(ph).*H.th + sin(th).*cos(ph).*H.r;
H.y =  cos(ph).*H.ph + cos(th).*sin(ph).*H.th + sin(th).*sin(ph).*H.r;
H.z =                - sin(th).*         H.th + cos(th).*         H.r;
% -------------------------------------------------------------------------
% intensities |E|^2/|E_0|^2 and |H|^2/|H_0|^2
% -------------------------------------------------------------------------
I.E = squeeze(E.x.*conj(E.x) + E.y.*conj(E.y) + E.z.*conj(E.z));
I.H = squeeze(H.x.*conj(H.x) + H.y.*conj(H.y) + H.z.*conj(H.z));
% -------------------------------------------------------------------------
end
function S = far_fld(l, T, th)
%FAR_FLD calculates polarized scattering waves S
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% l  - numbers of terms in expansion (array or scalar)
% T  - transfer matrix
% th - scattering angles theta
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% S - polarized scattering waves S:
%     total summation parallel and perpendicular to the scattering plane:
%           .par and .per
%     each mode l for parallel and perpendicular to the scattering plane:
%           .parl and .perl
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
cth = cos(th);                                                              % cosine theta for Pl
sth = sqrt(1 - cth.^2);                                                     % sinus theta
lpl   = [l,l(end)+1];                                                       % (l + 1) array
Pl  = zeros(numel(lpl),numel(th));
% -------------------------------------------------------------------------
%% CALCULATING ASSOCIATED LEGENDRE POLYNOMIALS AND THEIR DERIVATIVES
% -------------------------------------------------------------------------
for il = 1 : numel(lpl)
    tmpPl = legendre( lpl(il), cth ); Pl(il,:) = tmpPl(2,:);
end
% -------------------------------------------------------------------------
Pl = Pl./repmat(sth,numel(l)+1,1);
Pl(:,th == 0) = -lpl.*(lpl+1)./2;
% -------------------------------------------------------------------------
dPl = repmat(l',1,numel(cth)).*Pl(2:end,:) ...                              % recurrence relation
       - repmat((l+1)',1,numel(cth)).*repmat(cth,numel(l),1).*Pl(1:end-1,:);
Pl = Pl(1:end-1,:);
% -------------------------------------------------------------------------
%% CONSTRUCTING OUTPUT
% -------------------------------------------------------------------------
coef = repmat((2.*l + 1)./(l + 1)./l,numel(th),1)';
% -------------------------------------------------------------------------
a = -repmat((squeeze(T.te(:,end,2,1))./squeeze(T.te(:,end,1,1))),1,numel(th));
b = -repmat((squeeze(T.tm(:,end,2,1))./squeeze(T.tm(:,end,1,1))),1,numel(th));
% -------------------------------------------------------------------------
S.parl = coef.*(a.*dPl + b.* Pl);      S.perl = coef.*(a.* Pl + b.*dPl);
% -------------------------------------------------------------------------
S.par  = sum(S.parl,1);                S.per  = sum(S.perl,1);
% -------------------------------------------------------------------------
end
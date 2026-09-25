function F = F_lossy( A, B, x, bg, jl, hl, l_max )
%F_LOSSY calculates function F, which enters final expressions
%           for the total energy within lossy shells
% -------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------
% A, B  - expansion coefficients
% x     - size parameter
% bg    - indices of the lossy shells
% jl    - spherical Bessel functions of the 1st kind
% hl    - spherical Hankel functions of the 1st kind
% l_max - truncation for the l
% -------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------
% F - structure with functions F.xyyyz with the following notation:
%   x: 
%       - 'e' for electric
%       - 'm' for magnetic
%   yyy:
%       - 'min' for l-1 order
%       - 'sam' for l   order
%       - 'pls' for l+1 order
%   z:
%       - 'u' for upper boundary of definitive integral
%       - 'b' for lower boundary of definitive integral
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
% Author:        Ilia Rasskazov
%                irasskaz@ur.rochester.edu
% -------------------------------------------------------------------------
% Organization:  The Institute of Optics, University of Rochester
%                http://www.hajim.rochester.edu/optics/
% -------------------------------------------------------------------------
%% CALCULATING USEFUL VALUES
% -------------------------------------------------------------------------
xu = repmat( x.u(bg), l_max, 1 );
xb = repmat( x.b(bg), l_max, 1 );
% -------------------------------------------------------------------------
%% CONSTRUCTING OUTPUT
% -------------------------------------------------------------------------
F.eminu = 2*1i*imag( xu.*conj( A.e(:,bg).*jl.u(1:end-3,bg) + B.e(:,bg).*hl.u(1:end-3,bg) ).* ...
                             ( A.e(:,bg).*jl.u(2:end-2,bg) + B.e(:,bg).*hl.u(2:end-2,bg) ) );
F.mminu = 2*1i*imag( xu.*conj( A.m(:,bg).*jl.u(1:end-3,bg) + B.m(:,bg).*hl.u(1:end-3,bg) ).* ...
                             ( A.m(:,bg).*jl.u(2:end-2,bg) + B.m(:,bg).*hl.u(2:end-2,bg) ) );
% -------------------------------------------------------------------------
F.esamu = 2*1i*imag( xu.*conj( A.e(:,bg).*jl.u(2:end-2,bg) + B.e(:,bg).*hl.u(2:end-2,bg) ).* ...
                             ( A.e(:,bg).*jl.u(3:end-1,bg) + B.e(:,bg).*hl.u(3:end-1,bg) ) );
F.msamu = 2*1i*imag( xu.*conj( A.m(:,bg).*jl.u(2:end-2,bg) + B.m(:,bg).*hl.u(2:end-2,bg) ).* ...
                             ( A.m(:,bg).*jl.u(3:end-1,bg) + B.m(:,bg).*hl.u(3:end-1,bg) ) );
% -------------------------------------------------------------------------
F.eplsu = 2*1i*imag( xu.*conj( A.e(:,bg).*jl.u(3:end-1,bg) + B.e(:,bg).*hl.u(3:end-1,bg) ).* ...
                             ( A.e(:,bg).*jl.u(4:end  ,bg) + B.e(:,bg).*hl.u(4:end  ,bg) ) );
F.mplsu = 2*1i*imag( xu.*conj( A.m(:,bg).*jl.u(3:end-1,bg) + B.m(:,bg).*hl.u(3:end-1,bg) ).* ...
                             ( A.m(:,bg).*jl.u(4:end  ,bg) + B.m(:,bg).*hl.u(4:end  ,bg) ) );
% -------------------------------------------------------------------------                             
F.eminb = 2*1i*imag( xb.*conj( A.e(:,bg).*jl.b(1:end-3,bg) + B.e(:,bg).*hl.b(1:end-3,bg) ).* ...
                             ( A.e(:,bg).*jl.b(2:end-2,bg) + B.e(:,bg).*hl.b(2:end-2,bg) ) );
F.mminb = 2*1i*imag( xb.*conj( A.m(:,bg).*jl.b(1:end-3,bg) + B.m(:,bg).*hl.b(1:end-3,bg) ).* ...
                             ( A.m(:,bg).*jl.b(2:end-2,bg) + B.m(:,bg).*hl.b(2:end-2,bg) ) );
% -------------------------------------------------------------------------
F.esamb = 2*1i*imag( xb.*conj( A.e(:,bg).*jl.b(2:end-2,bg) + B.e(:,bg).*hl.b(2:end-2,bg) ).* ...
                             ( A.e(:,bg).*jl.b(3:end-1,bg) + B.e(:,bg).*hl.b(3:end-1,bg) ) );
F.msamb = 2*1i*imag( xb.*conj( A.m(:,bg).*jl.b(2:end-2,bg) + B.m(:,bg).*hl.b(2:end-2,bg) ).* ...
                             ( A.m(:,bg).*jl.b(3:end-1,bg) + B.m(:,bg).*hl.b(3:end-1,bg) ) );
% -------------------------------------------------------------------------
F.eplsb = 2*1i*imag( xb.*conj( A.e(:,bg).*jl.b(3:end-1,bg) + B.e(:,bg).*hl.b(3:end-1,bg) ).* ...
                             ( A.e(:,bg).*jl.b(4:end  ,bg) + B.e(:,bg).*hl.b(4:end  ,bg) ) );
F.mplsb = 2*1i*imag( xb.*conj( A.m(:,bg).*jl.b(3:end-1,bg) + B.m(:,bg).*hl.b(3:end-1,bg) ).* ...
                             ( A.m(:,bg).*jl.b(4:end  ,bg) + B.m(:,bg).*hl.b(4:end  ,bg) ) );                             
% -------------------------------------------------------------------------                           
end
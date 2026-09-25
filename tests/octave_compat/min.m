function varargout = min(varargin)
% shim: MATLAB's min(A,[],'all')
if nargin == 3 && ischar(varargin{3}) && strcmp(varargin{3},'all')
  [varargout{1:builtin('max',nargout,1)}] = builtin('min', varargin{1}(:));
else
  [varargout{1:builtin('max',nargout,1)}] = builtin('min', varargin{:});
end
end

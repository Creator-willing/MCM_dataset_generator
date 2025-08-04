function y = OTFS_modulation(N,M,x,padlen,varargin)
% OTFS Modulation:  ISFFT + Heisenberg transform
% X1=ifft(x);
% X2=X1.';
% X3=fft(X1);
% X4=fft(X2);
% x the row of x resperesents the different subcarrier
% M number of subcarrier
% N number of symbol

X = fft(ifft(x).').'/sqrt(M/N); %%%ISFFT
y = ifft(X.')*sqrt(M); % Heisenberg transform
y = y(:);  

if isempty(varargin)
    padtype = 'CP';
else
    padtype = varargin{1};
end
% Add cyclic prefix/zero padding according to padtype
switch padtype
    case 'CP'
        % % CP before each OTFS column (like OFDM) then serialize
        y = [y(end-padlen+1:end,:); y];  % cyclic prefix
        y = y(:);                        % serialize
    case 'ZP'
        % Zeros after each OTFS column then serialize
        %N = size(x,2);
        y = [y; zeros(padlen,N)];    % zero padding
        y = y(:);                    % serialize
    case 'RZP'
        % Serialize then append OTFS symbol with zeros
        y = y(:);                    % serialize
        y = [y; zeros(padlen,1)];    % zero padding
    case 'RCP'
        % Reduced CP
        % Serialize then prepend cyclic prefix
        y = y(:);                        % serialize
        y = [y(end-padlen+1:end); y];    % cyclic prefix
    case 'NONE'
        y = y(:);                   % no CP/ZP
    otherwise
        error('Invalid pad type');
end
end


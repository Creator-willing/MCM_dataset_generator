function [S] = AFDM_modulation(X, c1, c2,cp_len)
% S = AFDMmod(X, c1, c2) AFDM modulation with defined chirp frequencies
%   Inputs:
%      - X  : N x T transmit symbol matrix
%      - c1 : central digital frequency of first AFDM chirp
%      - c2 : central digital frequency of second AFDM chirp
%   Output:
%      - S  : N x T matrix of AFDM modulated samples

sizeX = size(X);
N = sizeX(1);

L1 = diag(exp(-1j*2*pi*c1*((0:N-1).^2)));
L2 = diag(exp(-1j*2*pi*c2*((0:N-1).^2)));
F = dftmtx(N)/sqrt(N);

IA = L1'*F'*L2';

S = IA*X;
if cp_len ~= 0
    n = -cp_len:1:-1;
    Ro = exp(-1j.*2.*pi.*c1.*(N^2+2*N*n));
    Ro = repmat(Ro,sizeX(2),1)';
    S = [S(end-cp_len+1:end,:).*Ro; S];
end
    
S = S(:);

end

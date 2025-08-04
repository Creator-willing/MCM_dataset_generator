function [S] = GFDM_modulation(X, pulse_shape, rolloff, overlapping, cyclic_prefix_length)
% GFDM_modulation - Generalized Frequency Division Multiplexing
% Inputs:
%   - X                     : K x M transmit symbol matrix
%   - pulse_shape           : Filter type ('RC', 'RRC', or custom array)
%   - rolloff               : Roll-off factor (used for RRC filter)
%   - overlapping           : Overlapping factor of subcarriers
%   - cyclic_prefix_length  : Length of the cyclic prefix
% Output:
%   - S                     : GFDM modulated signal with cyclic prefix

% Parameters
K = size(X, 1); % Number of subcarriers
M = size(X, 2); % Number of symbols per subcarrier
N = K * M;      % Total number of samples in the time domain

% Generate pulse shaping filter
if ischar(pulse_shape)
    if strcmp(pulse_shape, 'RC') % Rectangular filter
        pulse_shape = ones(1, M); % Rectangular pulse
    elseif strcmp(pulse_shape, 'RRC') % Root Raised Cosine filter
        span = 4; % Filter span in symbols
        sps = 1;  % Samples per symbol
        pulse_shape = rcosdesign(rolloff, span, sps, 'sqrt'); % RRC filter
    else
        error('Unsupported pulse shape type. Use "RC", "RRC", or provide custom filter.');
    end
end

% Normalize the filter energy
pulse_shape = pulse_shape / norm(pulse_shape);

% Extend the pulse shaping filter with overlapping
filter_length = length(pulse_shape);
pulse_shape_extended = repmat(pulse_shape, 1, overlapping);

% Ensure the signal length matches the number of samples
gfdm_signal = zeros(1, N + filter_length - 1);

% Generate the GFDM modulated signal
for k = 0:K-1
    for m = 0:M-1
        % Calculate the index for circular shift
        shift_amount = mod(k*M + m, N + filter_length - 1);
        % Circularly shift the pulse shaping filter
        g_shifted = circshift(pulse_shape_extended, shift_amount);

        % Ensure g_shifted is the same length as gfdm_signal
        g_shifted = g_shifted(1:length(gfdm_signal));

        % Contribute to the transmit signal
        gfdm_signal = gfdm_signal + X(k+1, m+1) * g_shifted;
    end
end

% Add cyclic prefix
if cyclic_prefix_length > 0
    cyclic_prefix = gfdm_signal(end-cyclic_prefix_length+1:end); % Extract CP
    S = [cyclic_prefix, gfdm_signal]; % Add CP to the signal
else
    S = gfdm_signal; % No CP
end

end
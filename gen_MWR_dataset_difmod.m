%% Function of this program: Generate MWR modulation recognition dataset
% Generate a multi-carrier modulation dataset, including "xAFDM", "xGFDM", 
% "xFOFDM", "xUFMC", "xFBMC", "xOTFS", "xOFDM", "xWOLA"
% For each type of multi-carrier modulation, three types of subcarrier modulation are included (QPSK, 16QAM, ...)
%% Input:
%% Output: MWR_all_AMC_diffmod.mat
%% Author: Mingkun Li
%% Date: 2025/1/4
clc 
clear all 
close all 


%% Parameter setting
% Sampling rate: should approximately match the power delay profile of the 
% channel and must be larger than "SubcarrierSpacing*NumberOfSubcarriers".
SamplingRate                = 15e3*256;
fs = SamplingRate;  % Sampling rate
dt                          = 1/SamplingRate;

% Simulation parameters
Simulation_SNR_OFDM_dB            = -18:2:20;                          % SNR for OFDM in dB. The average transmit power of all methods is the same! However, the SNR might be different due to filtering (in FOFDM and UFMC) or because a different bandwidth is used (different subcarrier spacing or different number of subcarriers).
Simulation_MonteCarloRepetitions  = 1800;                                % Number of Monte Carlo repetitions over which we take the average                  
Simulation_PlotSignalPowerTheory  = true;                               % If true, plot also the expected transmit power over time (theory)
snr_num  = length(Simulation_SNR_OFDM_dB);
SubcarrierSpacing          = 15e3;                                   % Subcarrier spacing (Hz)

subcarrier_mod_order = [4,16,64];  % QPSK,16QAM ... (order of subcarrier modulation)
sample_num_per_scmod = Simulation_MonteCarloRepetitions/length(subcarrier_mod_order);  % the number of samples of different subcarrier modulation

% Channel parameters
Channel_Velocity_kmh         = 60;                                     % Velocity in km/h. Note that [mph]*1.6=[kmh] and [m/s]*3.6=[kmh]
Channel_PowerDelayProfile    = 'VehicularA';                            % Power delay profile, either string or vector: 'Flat', 'AWGN', 'PedestrianA', 'PedestrianB', 'VehicularA', 'VehicularB', 'ExtendedPedestrianA', 'ExtendedPedestrianB', or 'TDL-A_xxns','TDL-B_xxns','TDL-C_xxns' (with xx the RMS delay spread in ns, e.g. 'TDL-A_30ns'), or [1 0 0.2] (Self-defined power delay profile which depends on the sampling rate)
Channel_DopplerModel         = 'Jakes';                                 % Which Doppler model: 'Jakes', 'Uniform', 'Discrete-Jakes', 'Discrete-Uniform'. For "Discrete-", we assume a discrete Doppler spectrum to improve the simulation time. This however only works accuratly if the number of samples and the velocity is sufficiently high.                                       
Channel_CarrierFrequency     = 1e9;                                     % Carrier Frequency (Hz)
normalized_freq_offset = 0.4;                                           % Normalized frequency offset
freq_offset = normalized_freq_offset * SamplingRate;                    % Absolute frequency offset (Hz)
pnoise_linewidth = 200e3;                                               % Phase noise linewidth (Hz)
M_NormalizedTimeOffset       = 0.05;                                     % Normalized time offset: t_off/T0=t_off*F

% General modulation paramters
QAM_ModulationOrder          = 4;                                      % QAM sigal constellation order: 4, 16, 64, 256, 1024,...

% FBMC parameters
FBMC_NumberOfSubcarriers     = 128;                                      % Number of subcarriers
FBMC_NumberOfSymbolsInTime   = 1;                                      % Number FBMC symbols in time
FBMC_SubcarrierSpacing       = SubcarrierSpacing;                                    % Subcarrier spacing (Hz)
FBMC_PrototypeFilter         = 'PHYDYAS-OQAM';                          % Prototype filter (Hermite, PHYDYAS, RRC) and OQAM or QAM.
FBMC_OverlappingFactor       = 4;                                       % Overlapping factor, 2,3,4,...

% OFDM parameters
OFDM_NumberOfSubcarriers     = 256;                                      % Number of subcarriers 
OFDM_NumberOfSymbolsInTime   = 4;                                      % Number OFDM symbols in time
OFDM_SubcarrierSpacing       = SubcarrierSpacing;                                    % Subcarrier spacing (Hz)
OFDM_CyclicPrefixLength      = 1/(8*OFDM_SubcarrierSpacing);           % Length of the cyclic prefix (s)

% WOLA parameters
WOLA_NumberOfSubcarriers     = 256;                                      % Number of subcarriers                                   
WOLA_NumberOfSymbolsInTime   = 4;                                      % Number WOLA symbols in time  
WOLA_SubcarrierSpacing       = SubcarrierSpacing;                                    % Subcarrier spacing (Hz)
WOLA_CyclicPrefixLength      = 1/(8*OFDM_SubcarrierSpacing);                                       % Length of the cyclic prefix (s) to combat the channel
WOLA_WindowLengthTX          = 1/(4*2*WOLA_SubcarrierSpacing);         % Length of the window overlapping (s) at the transmitter 
WOLA_WindowLengthRX          = 1/(4*2*WOLA_SubcarrierSpacing);         % Length of the window overlapping (s) at the receiver

% FOFDM parameters
FOFDM_NumberOfSubcarriers     = 256;                                     % Number of subcarriers
FOFDM_NumberOfSymbolsInTime   = 4;                                     % Number FOFDM symbols in time                        
FOFDM_SubcarrierSpacing       = SubcarrierSpacing;                                   % Subcarrier spacing (Hz)
FOFDM_CyclicPrefixLength      = 1/(8*OFDM_SubcarrierSpacing);                                      % Length of the cyclic prefix (s) to combat the channel
FOFDM_FilterLengthTX          = 0.2*1/(FOFDM_SubcarrierSpacing);        % Length of the transmit filter (s)
FOFDM_FilterLengthRX          = 0.2*1/(FOFDM_SubcarrierSpacing);        % Length of the receive filter (s) 
FOFDM_FilterCylicPrefixLength = 1/(4*FOFDM_SubcarrierSpacing);         % Length of the additional cyclic prefix (s) to combat ISI and ICI due to the filtering

% UFMC parameters
UFMC_NumberOfSubcarriers     = 256;                                      % Number of subcarriers
UFMC_NumberOfSymbolsInTime   = 4;                                      % Number UFMC symbols in time
UFMC_SubcarrierSpacing       = SubcarrierSpacing;                                    % Subcarrier spacing (Hz)
UFMC_CyclicPrefixLength      = 1/(8*OFDM_SubcarrierSpacing);                                       % Length of the cyclic prefix (s) to combat the channel. If zero padding is used, this length reprents the zero guard length instead of the CP length.
UFMC_FilterLengthTX          = 1/4*1/(UFMC_SubcarrierSpacing);         % Length of the transmit filter (s)
UFMC_FilterLengthRX          = 1/4*1/(UFMC_SubcarrierSpacing);         % Length of the receive filter (s)
UFMC_FilterCylicPrefixLength = 1/(4*UFMC_SubcarrierSpacing);           % Length of the additional cyclic prefix (or zero guard symbol if ZP is used) in seconds (s). Needed to combat ISI and ICI due to the filtering. However, small ICI and ISI is perfectly feasibly.
UFMC_ZeroPaddingInsteadOfCP  = true;                                    % TRUE for Zero Padding (ZP) and FALSE for a conventional Cyclic Prefix (CP)

% OTFS parpmeters
OTFS_NumberOfSubcarriers  = 32;                                         % number of subcarriers
OTFS_NumberOfSymbolsInTime = 32;                                        % number of subsymbols/frame
OTFS_padLen = 32;     % make this larger than the channel delay spread channel in samples
OTFS_padType = 'CP';  % this example requires ZP for ISI mitigation
OTFS_M_bits = log2(QAM_ModulationOrder);         % How many bits of information does each symbol carry

% AFDM parpmeters
AFDM_NumberOfSubcarriers  =   256;  % number of subcarriers
AFDM_guardWidth = 2;             % Guard width of the AFDM waveform
AFDM_NT = 4;                     % Symbol number
maxVel  = Channel_Velocity_kmh/3.6;   % Max Velocity（m/s）
maxDoppler = maxVel/(physconst("lightspeed")/Channel_CarrierFrequency)*2/SamplingRate*AFDM_NumberOfSubcarriers;
c1_AFDM = (2*(maxDoppler + AFDM_guardWidth) + 1)/(2*AFDM_NumberOfSubcarriers); % Satisfying the orthogonality condition
c2_AFDM = 1/(AFDM_NumberOfSubcarriers^2*pi);                              % Sufficiently small irrational number
%c1_AFDM = 1/(2*AFDM_NumberOfSubcarriers);
%c2_AFDM = 1/(2*AFDM_NT);
AFDM_M_bits = log2(QAM_ModulationOrder);         % How many bits of information does each symbol carry
AFDM_cp_len = 32;                        % length of CP


% SignalCarrier parpmeters 
SC_M_bits = log2(QAM_ModulationOrder);         % How many bits of information does each symbol carry
SC_N_syms_perfram = 1024;        % symbol number

% GFDM parpmeters
GFDM_NumberOfSubcarriers  = 32;  % number of subcarriers
GFDM_NumberOfSymbolsInTime = 32; % number of subsymbols/frame
GFDM_pulse_shape = 'rc';       % Filter type ('rc', 'rrc', or  'dirichlet' 'rect_fd' 'rect_td')
GFDM_rolloff = 0.1;            % Roll-off factor (used for RRC filter)
GFDM_overlapping = 2;          % Overlapping factor of subcarriers
GFDM_cyclic_prefix_length = 32; % Length of the cyclic prefix

%% Modulator and channel
% Generate " +Modulation\" Objects
% FBMC Object
FBMC = Modulation.FBMC(...
    FBMC_NumberOfSubcarriers,...                                        % Number of subcarriers
    FBMC_NumberOfSymbolsInTime,...                                      % Number FBMC symbols in time
    FBMC_SubcarrierSpacing,...                                          % Subcarrier spacing (Hz)
    SamplingRate,...                                                    % Sampling rate (Samples/s)
    0,...                                                               % Intermediate frequency of the first subcarrier (Hz).  Must be a multiple of the subcarrier spacing
    false,...                                                           % Transmit real valued signal (sampling theorem must be fulfilled!)
    FBMC_PrototypeFilter,...                                            % Prototype filter (Hermite, PHYDYAS, RRC) and OQAM or QAM. The data rate of QAM is reduced by a factor of two compared to OQAM, but robustness in doubly-selective channels is inceased
    FBMC_OverlappingFactor, ...                                         % Overlapping factor (also determines oversampling in the frequency domain)                                   
    0, ...                                                              % Initial phase shift
    true ...                                                            % Polyphase implementation
    );
FBMC_BlockOverlapTime = (FBMC.PrototypeFilter.OverlappingFactor-1/2)*FBMC.PHY.TimeSpacing;

% OFDM Object
OFDM = Modulation.OFDM(...
    OFDM_NumberOfSubcarriers,...                                        % Number of subcarriers
    OFDM_NumberOfSymbolsInTime,...                                      % Number OFDM symbols in time                                                 
    OFDM_SubcarrierSpacing,...                                          % Subcarrier spacing (Hz) 
    SamplingRate,...                                                    % Sampling rate (Samples/s)                                       
    0,...                                                               % Intermediate frequency of the first subcarrier (Hz). Must be a multiple of the subcarrier spacing
    false,...                                                           % Transmit real valued signal (sampling theorem must be fulfilled!)
    OFDM_CyclicPrefixLength, ...                                        % Length of the cyclic prefix (s)                 
    0 ...                                           % Length of the guard time (s), that is, zeros at the beginning and at the end of the transmission
    );

% Windowed OFDM (WOLA)
WOLA = Modulation.WOLA(...
    WOLA_NumberOfSubcarriers,...                                        % Number subcarriers
    WOLA_NumberOfSymbolsInTime,...                                      % Number WOLA symbols in time 
    WOLA_SubcarrierSpacing,...                                          % Subcarrier spacing (Hz)
    SamplingRate,...                                                    % Sampling rate (Samples/s)
    0,...                                                               % Intermediate frequency of the first subcarrier (Hz). Must be a multiple of the subcarrier spacing
    false,...                                                           % Transmit real valued signal (sampling theorem must be fulfilled!)
    0, ...                                                              % Length of the cyclic prefix (s)
    0, ...                    % Length of the guard time (s), that is, zeros at the beginning and at the end of the transmission
    WOLA_WindowLengthTX, ...                                            % Length of the window overlapping (s) at the transmitter 
    WOLA_WindowLengthRX ...                                             % Length of the window overlapping (s) at the receiver
    );

% FOFDM (Filtered OFDM)
FOFDM = Modulation.FOFDM(...
    FOFDM_NumberOfSubcarriers,...                                       % Number of subcarriers
    FOFDM_NumberOfSymbolsInTime,...                                     % Number FOFDM symbols in time                
    FOFDM_SubcarrierSpacing,...                                         % Subcarrier spacing (Hz)
    SamplingRate,...                                                    % Sampling rate (Samples/s)
    0,...                                                               % Intermediate frequency of the first subcarrier (Hz). Must be a multiple of the subcarrier spacing
    false,...                                                           % Transmit real valued signal (sampling theorem must be fulfilled!)
    0, ...                                                              % Length of the cyclic prefix (s)
    0, ...                   % Length of the guard time (s), that is, zeros at the beginning and at the end of the transmission                    
    FOFDM_FilterLengthTX, ...                                           % Length of the transmit filter (s)
    FOFDM_FilterLengthRX, ...                                           % Length of the receive filter (s) 
    FOFDM_FilterCylicPrefixLength ...                                   % Length of the additional cyclic prefix (s).  Needed to combat ISI and ICI due to the filtering. However, some small ICI and ISI is perfectly fine.
);

% UFMC (Subband Filtered OFDM)
UFMC = Modulation.UFMC(...
    UFMC_NumberOfSubcarriers,...                                        % Number of subcarriers
    UFMC_NumberOfSymbolsInTime,...                                      % Number UFMC symbols in time
    UFMC_SubcarrierSpacing,...                                          % Subcarrier spacing (Hz)
    SamplingRate,...                                                    % Sampling rate (Samples/s)
    0,...                                                               % Intermediate frequency of the first subcarrier (Hz). Must be a multiple of the subcarrier spacing
    false,...                                                           % Transmit real valued signal (sampling theorem must be fulfilled!)
    0, ...                                                              % Length of the cyclic prefix (s). If zero padding is used, this length reprents the zero guard length instead of the CP length
    0, ...                    % Length of the guard time (s), that is, zeros at the beginning and at the end of the transmission
    UFMC_FilterLengthTX, ...                                            % Length of the transmit filter (s)
    UFMC_FilterLengthRX, ...                                            % Length of the receive filter (s)
    UFMC_FilterCylicPrefixLength, ...                                   % Length of the additional cyclic prefix (or zero guard symbol if ZP is used) in seconds (s). Needed to combat ISI and ICI due to the filtering. However, some small ICI and ISI is perfectly fine.
    UFMC_ZeroPaddingInsteadOfCP ...                                     % TRUE for Zero Padding (ZP) and FALSE for a conventional Cyclic Prefix (CP)
);

% Number of samples
N_FBMC  = FBMC.Nr.SamplesTotal;
N_OFDM  = OFDM.Nr.SamplesTotal;
N_WOLA  = WOLA.Nr.SamplesTotal;
N_FOFDM = FOFDM.Nr.SamplesTotal;
N_UFMC  = UFMC.Nr.SamplesTotal;
N_SC    = SC_N_syms_perfram;
N_OTFS = (OTFS_NumberOfSubcarriers + OTFS_padLen) * OTFS_NumberOfSymbolsInTime ;
N_AFDM = (AFDM_NumberOfSubcarriers+AFDM_cp_len)*AFDM_NT;
N_GFDM = GFDM_NumberOfSubcarriers * GFDM_NumberOfSymbolsInTime;
N       = max([N_FBMC N_OFDM N_WOLA N_FOFDM N_UFMC N_OTFS N_AFDM]);

ChannelModel = Channel.FastFading(...
    SamplingRate,...                                                    % Sampling rate (Samples/s)
    Channel_PowerDelayProfile,...                                       % Power delay profile, either string or vector: 'Flat', 'AWGN', 'PedestrianA', 'PedestrianB', 'VehicularA', 'VehicularB', 'ExtendedPedestrianA', 'ExtendedPedestrianB', or 'TDL-A_xxns','TDL-B_xxns','TDL-C_xxns' (with xx the RMS delay spread in ns, e.g. 'TDL-A_30ns'), or [1 0 0.2] (Self-defined power delay profile which depends on the sampling rate) 
    N,...                                                               % Number of total samples
    Channel_Velocity_kmh/3.6*Channel_CarrierFrequency/2.998e8,...       % Maximum Doppler shift: Velocity_kmh/3.6*CarrierFrequency/2.998e8  
    Channel_DopplerModel,...                                            % Which Doppler model: 'Jakes', 'Uniform', 'Discrete-Jakes', 'Discrete-Uniform'. For "Discrete-", we assume a discrete Doppler spectrum to improve the simulation time. This only works accuratly if the number of samples and the velocity is sufficiently large                                       
    5, ...                                                            % Number of paths for the WSSUS process. Only relevant for a 'Jakes' and 'Uniform' Doppler spectrum                                                 
    1,...                                                               % Number of transmit antennas
    1,...                                                               % Number of receive antennas
    true ...                                                            % Gives a warning if the predefined delay taps of the channel do not fit the sampling rate. This is usually not much of a problem if they are approximatly the same.
    );



modulator_list = cell(2,length(subcarrier_mod_order));
for i = 1:length(subcarrier_mod_order)
    temp_order = subcarrier_mod_order(i);
    % PAM and QAM Object
    QAM = Modulation.SignalConstellation(temp_order,'QAM');
    if strcmp(FBMC.Method(end-3),'O')
        % FBMC-OQAM transmission, only real valued data symbols
        PAMorQAM = Modulation.SignalConstellation(sqrt(temp_order),'PAM');
    else
        % FBMC-QAM transmission,  complex valued data symbols
        PAMorQAM = Modulation.SignalConstellation(temp_order,'QAM');
    end
    modulator_list{1,i} = QAM;
    modulator_list{2,i} = PAMorQAM;
end




%% Generate dataset
mod_list = ["xAFDM","xGFDM","xFOFDM","xUFMC","xFBMC","xOTFS","xOFDM","xWOLA"];
mod_num = length(mod_list);
for mod_index = 1:length(mod_list)
tic
save_len = 1024;

temp_mod_data = zeros(snr_num,Simulation_MonteCarloRepetitions,2,save_len);

mod_name = mod_list(mod_index);

for i_SNR = 1:length(Simulation_SNR_OFDM_dB)
    temp_gen_data = zeros(Simulation_MonteCarloRepetitions,2,save_len);
    % Add noise
    SNR_OFDM_dB = Simulation_SNR_OFDM_dB(i_SNR);
    snr = SNR_OFDM_dB;
    Pn_time     = 1/OFDM.GetSymbolNoisePower(1)*10^(-SNR_OFDM_dB/10);
    
for i_rep = 1:Simulation_MonteCarloRepetitions
    % Obtain the current subcarrier modulation order
    sc_mod_index = floor((i_rep-1)/sample_num_per_scmod)+1;
    QAM_ModulationOrder = subcarrier_mod_order(sc_mod_index); % Subcarrier modulation order
    QAM = modulator_list{1,sc_mod_index} ;    % modulator
    PAMorQAM = modulator_list{2,sc_mod_index};% modulator
    M_bits = log2(QAM_ModulationOrder);         % How many bits of information does each symbol carry


    % Update channel
    ChannelModel.NewRealization;
    
    if mod_name == "xFBMC"
        BinaryDataStream_FBMC   = randi([0 1], FBMC.Nr.Subcarriers  * FBMC.Nr.MCSymbols  * log2(PAMorQAM.ModulationOrder),1);
        x_FBMC  = reshape( PAMorQAM.Bit2Symbol(BinaryDataStream_FBMC)  , FBMC.Nr.Subcarriers  , FBMC.Nr.MCSymbols);
        s_moded  =  FBMC.Modulation( x_FBMC ); 
    elseif mod_name == "xOFDM"
        %N_sig = N_OFDM;
        BinaryDataStream_OFDM   = randi([0 1], OFDM.Nr.Subcarriers  * OFDM.Nr.MCSymbols  * log2(QAM.ModulationOrder),1);
        x_OFDM  = reshape(      QAM.Bit2Symbol(BinaryDataStream_OFDM)  , OFDM.Nr.Subcarriers  , OFDM.Nr.MCSymbols);
        s_moded  =  OFDM.Modulation( x_OFDM );
    elseif mod_name == "xWOLA"
        %N_sig = N_WOLA;
        BinaryDataStream_WOLA   = randi([0 1], WOLA.Nr.Subcarriers  * WOLA.Nr.MCSymbols  * log2(QAM.ModulationOrder),1);
        x_WOLA  = reshape(      QAM.Bit2Symbol(BinaryDataStream_WOLA)  , WOLA.Nr.Subcarriers  , WOLA.Nr.MCSymbols);
        s_moded  =  WOLA.Modulation( x_WOLA );
    elseif mod_name == "xFOFDM"
        BinaryDataStream_FOFDM  = randi([0 1], FOFDM.Nr.Subcarriers * FOFDM.Nr.MCSymbols * log2(QAM.ModulationOrder),1);
        x_FOFDM = reshape(      QAM.Bit2Symbol(BinaryDataStream_FOFDM) , FOFDM.Nr.Subcarriers , FOFDM.Nr.MCSymbols);
        s_moded = FOFDM.Modulation( x_FOFDM );
    elseif mod_name == "xUFMC"
        BinaryDataStream_UFMC   = randi([0 1], UFMC.Nr.Subcarriers * UFMC.Nr.MCSymbols * log2(QAM.ModulationOrder),1);
        x_UFMC  = reshape(      QAM.Bit2Symbol(BinaryDataStream_UFMC)  , UFMC.Nr.Subcarriers  , UFMC.Nr.MCSymbols);
        s_moded  =  UFMC.Modulation( x_UFMC );
    elseif mod_name == "xSC"
        BinaryDataStream_SC     = randi([0 1], M_bits * SC_N_syms_perfram,1);
        x_SC = BinaryDataStream_SC;
        s_moded = qammod(reshape(x_SC,M_bits,[]), QAM_ModulationOrder,'gray','InputType','bit', 'UnitAveragePower', true);
        s_moded = s_moded';
    elseif mod_name == "xOTFS"
        BinaryDataStream_OTFS     = randi([0 1], M_bits * OTFS_NumberOfSubcarriers * OTFS_NumberOfSymbolsInTime,1);
        x_UFMC  = reshape( QAM.Bit2Symbol(BinaryDataStream_OTFS), OTFS_NumberOfSymbolsInTime, OTFS_NumberOfSubcarriers);
        s_moded = OTFS_modulation(OTFS_NumberOfSymbolsInTime,OTFS_NumberOfSubcarriers,x_UFMC,OTFS_padLen,OTFS_padType);
    elseif mod_name == "xAFDM"
        BinaryDataStream_AFDM     = randi([0 1], M_bits * AFDM_NumberOfSubcarriers * AFDM_NT,1);
        x_AFDM = reshape( QAM.Bit2Symbol(BinaryDataStream_AFDM),  AFDM_NumberOfSubcarriers,AFDM_NT);
        s_moded = AFDM_modulation(x_AFDM,c1_AFDM,c2_AFDM,AFDM_cp_len);
    elseif mod_name == "xGFDM"
        [GFDM_Tx_real, GFDM_Tx_imag] = GFDM_Coder(GFDM_NumberOfSubcarriers,GFDM_NumberOfSymbolsInTime,GFDM_pulse_shape,GFDM_rolloff,GFDM_overlapping,GFDM_cyclic_prefix_length,M_bits);
        s_moded = (GFDM_Tx_real + 1j * GFDM_Tx_imag)*sqrt(10);
    end
    N_sig = length(s_moded);
    % P = sum(s_moded .* conj(s_moded))/length(s_moded)


    % Add frequency offset
    t = (0:N_sig-1)' / fs; % Time vector
    freq_offset_signal = s_moded .* exp(1j * 2 * pi * freq_offset * SubcarrierSpacing * t);

    % Add time offset
    M_NormalizedTimeOffset_Samples      = round((M_NormalizedTimeOffset/SubcarrierSpacing)/dt);
    M_NormalizedTimeOffset_Samples = randi(M_NormalizedTimeOffset_Samples);
    time_offset_signal = [zeros(M_NormalizedTimeOffset_Samples,1); freq_offset_signal(1:end-M_NormalizedTimeOffset_Samples)];
    

    % Add phase noise
    %phase_noise = cumsum(sqrt(2 * pi * pnoise_linewidth / fs) * randn(N_sig, 1));
    %tx_signal_with_pnoise = time_offset_signal .* exp(1j * phase_noise);
    tx_signal_with_pnoise = time_offset_signal;
    

    % Channel convolution
    r_noNoise  = ChannelModel.Convolution( tx_signal_with_pnoise );
    
    %r_noNoise = s_moded;
    % Add noise
    N = max(N,N_sig);
    noise   = sqrt(Pn_time/2)*(randn(N,1)+1j*randn(N,1));
    rxSig = r_noNoise  + noise(1:N_sig);
    %rxSig = s_moded;

    % save data
    temp_gen_data(i_rep,1,:) = real(rxSig(1:save_len));
    temp_gen_data(i_rep,2,:) = imag(rxSig(1:save_len));

    %
    if mod(i_rep,100) == 0
        clc
        fprintf('MCM-(mod,SNR)');disp([mod_name,num2str(snr)]);
        display(i_rep,'Number of frames');
    end

end
    % Data normalization
    temp_gen_data = (temp_gen_data-mean(temp_gen_data,'all'))./std(temp_gen_data,0,'all');
    temp_mod_data(i_SNR,:,:,:) = temp_gen_data;
    
end 
    assignin('base', mod_list(mod_index), temp_mod_data);
    % save data as ".mat" form localy
    if mod_index == 1
        save('MWR_all_AMC_diffmod.mat',mod_list(mod_index))
    else
        save('MWR_all_AMC_diffmod.mat',mod_list(mod_index),"-append"); 
    end
end

%% show data
figure(1)
% Display the real part of the data
subplot(3,6,1)
plot(squeeze(xGFDM(20,1,1,:)))
title('GFDM')
subplot(3,6,2)
plot(squeeze(xFOFDM(20,1,1,:)))
title('FOFDM')
subplot(3,6,3)
plot(squeeze(xUFMC(20,1,1,:)))
title('UFMC')
subplot(3,6,4)
plot(squeeze(xFBMC(20,1,1,:)))
title('FBMC')
subplot(3,6,5)
plot(squeeze(xOTFS(20,1,1,:)))
title('OTFS')
subplot(3,6,6)
plot(squeeze(xOFDM(20,1,1,:)))
title('OFDM')
% Display the imaginary part of the data
subplot(3,6,7)
plot(squeeze(xGFDM(20,1,2,:)))
title('GFDM')
subplot(3,6,8)
plot(squeeze(xFOFDM(20,1,2,:)))
title('FOFDM')
subplot(3,6,9)
plot(squeeze(xUFMC(20,1,2,:)))
title('UFMC')
subplot(3,6,10)
plot(squeeze(xFBMC(20,1,2,:)))
title('FBMC')
subplot(3,6,11)
plot(squeeze(xOTFS(20,1,2,:)))
title('OTFS')
subplot(3,6,12)
plot(squeeze(xOFDM(20,1,2,:)))
title('OFDM')
% Display the magnitude of the data
subplot(3,6,13)
plot(abs(squeeze(xGFDM(20,1,1,:))+1j*squeeze(xGFDM(20,1,2,:))))
title('GFDM')
subplot(3,6,14)
plot(abs(squeeze(xFOFDM(20,1,1,:))+1j*squeeze(xFOFDM(20,1,2,:))))
title('FOFDM')
subplot(3,6,15)
plot(abs(squeeze(xUFMC(20,1,1,:))+1j*squeeze(xUFMC(20,1,2,:))))
title('UFMC')
subplot(3,6,16)
plot(abs(squeeze(xFBMC(20,1,1,:))+1j*squeeze(xFBMC(20,1,2,:))))
title('FBMC')
subplot(3,6,17)
plot(abs(squeeze(xOTFS(20,1,1,:))+1j*squeeze(xOTFS(20,1,2,:))))
title('OTFS')
subplot(3,6,18)
plot(abs(squeeze(xOFDM(20,1,1,:))+1j*squeeze(xOFDM(20,1,2,:))))
title('OFDM')

# MCM Dataset Generator

Generate datasets for multicarrier modulation classification with comprehensive parameter configurations and multiple modulation schemes.

## Overview

This repository contains MATLAB scripts for generating Multi-Carrier Modulation (MCM) recognition datasets. Two main scripts are provided:

- **`gen_MWR_dataset.m`**: Standard MCM dataset with fixed QPSK modulation
- **`gen_MWR_dataset_difmod.m`**: Enhanced dataset with multiple subcarrier modulation schemes (QPSK, 16QAM, 64QAM)

## Dataset Generation Process

### Common Process Flow

1. **Parameter Initialization**
   - Set sampling rate, SNR range, Monte Carlo repetitions
   - Configure channel parameters (velocity, power delay profile, Doppler model)
   - Define modulation-specific parameters

2. **Modulator Object Creation**
   - Create FBMC, OFDM, WOLA, FOFDM, UFMC, OTFS, AFDM, GFDM modulator objects
   - Initialize channel model with FastFading

3. **Data Generation Loop**
   - For each modulation type
   - For each SNR level
   - For each Monte Carlo repetition:
     - Generate random binary data stream
     - Apply modulation scheme
     - Add impairments (frequency offset, time offset, phase noise)
     - Pass through channel model
     - Add AWGN noise
     - Save real and imaginary parts

4. **Data Processing**
   - Normalize data (zero mean, unit variance)
   - Save to .mat file

### Key Differences Between Scripts

| Aspect | `gen_MWR_dataset.m` | `gen_MWR_dataset_difmod.m` |
|--------|---------------------|----------------------------|
| **SNR Range** | -18:2:20 dB (20 SNR levels) | -18:2:20 dB (20 SNR levels) |
| **Monte Carlo Repetitions** | 2000 | 1800 |
| **Subcarrier Modulation** | Fixed QPSK (4-QAM) | Multiple: QPSK, 16QAM, 64QAM |
| **Modulation Distribution** | Equal across all modulations | 600 samples per subcarrier modulation |
| **Output File** | `MWR_all_AMC.mat` | `MWR_all_AMC_diffmod.mat` |

## Parameter Definitions

### System Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SamplingRate` | 15e3×256 Hz | System sampling rate |
| `fs` | 15e3×256 Hz | Sampling frequency |
| `dt` | 1/SamplingRate | Time step |
| `SubcarrierSpacing` | 15e3 Hz | Subcarrier spacing |

### Simulation Parameters

| Parameter | `gen_MWR_dataset.m` | `gen_MWR_dataset_difmod.m` | Description |
|-----------|---------------------|----------------------------|-------------|
| `Simulation_SNR_OFDM_dB` | -18:2:20 | -18:2:20 | SNR range in dB |
| `Simulation_MonteCarloRepetitions` | 2000 | 1800 | Number of Monte Carlo iterations |
| `snr_num` | 20 | 3 | Number of SNR levels |
| `subcarrier_mod_order` | N/A | [4,16,64] | Subcarrier modulation orders |
| `sample_num_per_scmod` | N/A | 600 | Samples per subcarrier modulation |

### Channel Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Channel_Velocity_kmh` | 60 | Vehicle velocity in km/h |
| `Channel_PowerDelayProfile` | 'VehicularA' | Power delay profile type |
| `Channel_DopplerModel` | 'Jakes' | Doppler spectrum model |
| `Channel_CarrierFrequency` | 1e9 Hz | Carrier frequency |
| `normalized_freq_offset` | 0.4 | Normalized frequency offset |
| `freq_offset` | normalized_freq_offset × SamplingRate | Absolute frequency offset |
| `pnoise_linewidth` | 200e3 Hz | Phase noise linewidth |
| `M_NormalizedTimeOffset` | 0.05 | Normalized time offset |

### Modulation Parameters

#### FBMC Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `FBMC_NumberOfSubcarriers` | 128 | Number of subcarriers |
| `FBMC_NumberOfSymbolsInTime` | 1 | Number of FBMC symbols |
| `FBMC_PrototypeFilter` | 'PHYDYAS-OQAM' | Prototype filter type |
| `FBMC_OverlappingFactor` | 4 | Overlapping factor |

#### OFDM Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `OFDM_NumberOfSubcarriers` | 256 | Number of subcarriers |
| `OFDM_NumberOfSymbolsInTime` | 4 | Number of OFDM symbols |
| `OFDM_CyclicPrefixLength` | 1/(8×OFDM_SubcarrierSpacing) | CP length |

#### WOLA Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `WOLA_NumberOfSubcarriers` | 256 | Number of subcarriers |
| `WOLA_NumberOfSymbolsInTime` | 4 | Number of WOLA symbols |
| `WOLA_WindowLengthTX/RX` | 1/(4×2×WOLA_SubcarrierSpacing) | Window length |

#### FOFDM Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `FOFDM_NumberOfSubcarriers` | 256 | Number of subcarriers |
| `FOFDM_NumberOfSymbolsInTime` | 4 | Number of FOFDM symbols |
| `FOFDM_FilterLengthTX/RX` | 0.2×1/FOFDM_SubcarrierSpacing | Filter length |

#### UFMC Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `UFMC_NumberOfSubcarriers` | 256 | Number of subcarriers |
| `UFMC_NumberOfSymbolsInTime` | 4 | Number of UFMC symbols |
| `UFMC_FilterLengthTX/RX` | 1/4×1/UFMC_SubcarrierSpacing | Filter length |
| `UFMC_ZeroPaddingInsteadOfCP` | true | Use ZP instead of CP |

#### OTFS Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `OTFS_NumberOfSubcarriers` | 32 | Number of subcarriers |
| `OTFS_NumberOfSymbolsInTime` | 32 | Number of subsymbols |
| `OTFS_padLen` | 32 | Padding length |
| `OTFS_padType` | 'CP' | Padding type |

#### AFDM Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `AFDM_NumberOfSubcarriers` | 256 | Number of subcarriers |
| `AFDM_guardWidth` | 2 | Guard width |
| `AFDM_NT` | 4 | Symbol number |
| `AFDM_cp_len` | 32 | CP length |

#### GFDM Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `GFDM_NumberOfSubcarriers` | 32 | Number of subcarriers |
| `GFDM_NumberOfSymbolsInTime` | 32 | Number of subsymbols |
| `GFDM_pulse_shape` | 'rc' | Filter type |
| `GFDM_rolloff` | 0.1 | Roll-off factor |
| `GFDM_overlapping` | 2 | Overlapping factor |
| `GFDM_cyclic_prefix_length` | 32 | CP length |

## Supported Modulation Schemes

| Modulation | Description | Implementation |
|------------|-------------|----------------|
| **xAFDM** | Affine Frequency Division Multiplexing | Custom AFDM_modulation function |
| **xGFDM** | Generalized Frequency Division Multiplexing | Custom GFDM_Coder function |
| **xFOFDM** | Filtered OFDM | Built-in FOFDM modulator |
| **xUFMC** | Universal Filtered Multi-Carrier | Built-in UFMC modulator |
| **xFBMC** | Filter Bank Multi-Carrier | Built-in FBMC modulator |
| **xOTFS** | Orthogonal Time Frequency Space | Custom OTFS_modulation function |
| **xOFDM** | Orthogonal Frequency Division Multiplexing | Built-in OFDM modulator |
| **xWOLA** | Windowed OFDM | Built-in WOLA modulator |

## Data Structure

The generated dataset has the following structure:
- **Dimensions**: `[SNR_levels, Monte_Carlo_repetitions, 2, 1024]`
- **Channel 1**: Real part of received signal
- **Channel 2**: Imaginary part of received signal
- **Signal length**: 1024 samples per frame
- **Data format**: Normalized (zero mean, unit variance)

## Impairment Models

| Impairment | Implementation | Parameters |
|------------|----------------|------------|
| **Frequency Offset** | `exp(1j * 2 * pi * freq_offset * SubcarrierSpacing * t)` | 0.4 normalized offset |
| **Time Offset** | Random delay with zeros padding | 0.05 normalized offset |
| **Phase Noise** | Commented out (not used) | 200e3 Hz linewidth |
| **Channel** | VehicularA with Jakes Doppler | 60 km/h velocity |
| **AWGN** | Complex Gaussian noise | SNR-dependent power |

## Usage

1. **Standard Dataset Generation**:
   ```matlab
   run('gen_MWR_dataset.m')
   ```

2. **Enhanced Dataset Generation** (with multiple subcarrier modulations):
   ```matlab
   run('gen_MWR_dataset_difmod.m')
   ```

## Output Files

- `MWR_all_AMC.mat`: Standard dataset with fixed QPSK modulation
- `MWR_all_AMC_diffmod.mat`: Enhanced dataset with multiple subcarrier modulations

## Dependencies

- MATLAB with Signal Processing Toolbox
- Custom modulation functions (AFDM_modulation, OTFS_modulation, GFDM_Coder)
- Channel estimation and modulation libraries in the `+Channel` and `+Modulation` folders

## Dataset Division
It is recommended to split the dataset in a 7:1:2 ratio. The generated file MWR_all_AMC.mat contains 20 SNR levels, 8 classes, and 2000 samples per class per SNR, resulting in a total of 320,000 samples. Among them, 70% are used for training, yielding 224,000 training samples. 20% are used for testing, resulting in 64,000 test samples.

## Simulation env of paper
the hardware and software configurations of the test environment
```
	Cuda device count:  1
	Cuda device:  NVIDIA GeForce RTX 4070 Laptop GPU
	Torch version:  2.4.1+cu124
	Cuda version:  12.4
	Cudnn version:  90100
	System：Windows11
	Programming language：Python3.12.7
	IDE: Visual Studio code
```

## Author

Mingkun Li  




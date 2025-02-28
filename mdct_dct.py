import numpy as np
import soundfile as sf
import scipy.fftpack as fft
import os

# Define Input & Output Paths
INPUT_FILES = [
    "data/quantized_24bit.wav",
    "data/quantized_16bit.wav",
    #"data/quantized_8bit.wav"
]
OUTPUT_DIR = "data/mdct_dct_outputs/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

'''# Apply MDCT (Modified Discrete Cosine Transform)
def apply_mdct(audio):
    """Applies MDCT transformation to an audio signal."""
    N = len(audio) // 2 * 2  # Ensure even length
    window = np.hanning(N)  # Apply Hann window
    mdct_coeffs = fft.dct(audio[:N] * window, type=2, norm='ortho')
    return mdct_coeffs'''

def apply_mdct(audio):
    """Applies MDCT transformation to an audio signal."""
    if len(audio.shape) == 2:  # Stereo audio
        N = audio.shape[0] // 2 * 2  # Ensure even length
        window = np.hanning(N)[:, np.newaxis]  # Add dimension for broadcasting
        mdct_coeffs = fft.dct(audio[:N] * window, type=2, norm='ortho')
        return mdct_coeffs
    else:  # Mono audio
        N = len(audio) // 2 * 2  # Ensure even length
        window = np.hanning(N)
        mdct_coeffs = fft.dct(audio[:N] * window, type=2, norm='ortho')
        return mdct_coeffs


'''# Apply MDCT on Subbands
def apply_mdct_subbands(audio, subband_size):
    """Applies MDCT to each subband of the given size."""
    num_subbands = len(audio) // subband_size
    mdct_coeffs = []
    
    for i in range(num_subbands):
        subband = audio[i * subband_size:(i + 1) * subband_size]
        window = np.hanning(subband_size)  # Hann window for smoothing
        mdct_coeffs.append(fft.dct(subband * window, type=2, norm='ortho'))
    
    return np.array(mdct_coeffs)'''


'''# Apply DCT (Discrete Cosine Transform)
def apply_dct(audio):
    """Applies DCT transformation to an audio signal."""
    dct_coeffs = fft.dct(audio, type=2, norm='ortho')
    return dct_coeffs'''

def apply_dct(audio):
    """Applies DCT transformation to an audio signal."""
    if len(audio.shape) == 2:  # Stereo audio
        dct_coeffs = fft.dct(audio, type=2, norm='ortho')
    else:  # Mono audio
        dct_coeffs = fft.dct(audio, type=2, norm='ortho')
    return dct_coeffs


'''# Apply DCT on Subbands
def apply_dct_subbands(audio, subband_size):
    """Applies DCT to each subband of the given size."""
    num_subbands = len(audio) // subband_size
    dct_coeffs = []
    
    for i in range(num_subbands):
        subband = audio[i * subband_size:(i + 1) * subband_size]
        dct_coeffs.append(fft.dct(subband, type=2, norm='ortho'))
    
    return np.array(dct_coeffs)'''

# Process Each Quantized File
for file in INPUT_FILES:
    if os.path.exists(file):
        audio, samplerate = sf.read(file)

        # Apply MDCT and DCT
        mdct_audio = apply_mdct(audio)
        dct_audio = apply_dct(audio)

        # Save transformed coefficients as .npy files
        mdct_output = os.path.join(OUTPUT_DIR, os.path.basename(file).replace(".wav", "_mdct.npy"))
        dct_output = os.path.join(OUTPUT_DIR, os.path.basename(file).replace(".wav", "_dct.npy"))
        
        np.save(mdct_output, mdct_audio)
        np.save(dct_output, dct_audio)

        print(f"Processed {file} - Saved MDCT & DCT outputs.")
    else:
        print(f"Error: {file} not found!")

print("MDCT & DCT Processing Completed!")

'''# Process Each Quantized File
for file in INPUT_FILES:
    if os.path.exists(file):
        audio, samplerate = sf.read(file)

        # Apply MDCT and DCT using subband sizes 64 and 120
        mdct_64 = apply_mdct_subbands(audio, 64)
        mdct_120 = apply_mdct_subbands(audio, 120)
        dct_64 = apply_dct_subbands(audio, 64)
        dct_120 = apply_dct_subbands(audio, 120)

        # Save transformed coefficients
        np.save(os.path.join(OUTPUT_DIR, os.path.basename(file).replace(".wav", "_mdct_64.npy")), mdct_64)
        np.save(os.path.join(OUTPUT_DIR, os.path.basename(file).replace(".wav", "_mdct_120.npy")), mdct_120)
        np.save(os.path.join(OUTPUT_DIR, os.path.basename(file).replace(".wav", "_dct_64.npy")), dct_64)
        np.save(os.path.join(OUTPUT_DIR, os.path.basename(file).replace(".wav", "_dct_120.npy")), dct_120)

        print(f"Processed {file} - MDCT & DCT applied on subbands of 64 and 120.")
    else:
        print(f"Error: {file} not found!")

print("MDCT & DCT Processing Completed with Subbands!")'''
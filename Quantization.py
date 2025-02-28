import numpy as np
import soundfile as sf
import os

#  Define Input & Output Paths
INPUT_FILE = "data/input_32bit.wav"  # Make sure this file exists!
OUTPUT_DIR = "data/"

#  Quantization Function
def quantize_audio(input_file, bit_depth):
    """Quantizes a 32-bit float WAV file to the specified bit depth."""
    
    # Load audio file
    audio, samplerate = sf.read(input_file, dtype='float32')

    # Define quantization levels
    max_val = 2 ** (bit_depth - 1) - 1  
    quantized_audio = np.round(audio * max_val) / max_val  # Apply quantization

    # Save the quantized file
    output_file = os.path.join(OUTPUT_DIR, f"quantized_{bit_depth}bit.wav")
    sf.write(output_file, quantized_audio, samplerate, subtype=f"PCM_{bit_depth}")

    print(f" Quantized to {bit_depth}-bit and saved as {output_file}")

#  Run Quantization for 24-bit, 16-bit
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f" Error: {INPUT_FILE} not found! Please add a valid 32-bit WAV file.")
    else:
        quantize_audio(INPUT_FILE, 24)
        quantize_audio(INPUT_FILE, 16)
        print(" Quantization Completed!")



'''psychoacoustic_model.py:

# Core functions for psychoacoustic modeling:
- compute_masking_threshold()
- compute_critical_bands()
- spreading_function()

Copy

Insert at cursor
python
subband_analysis.py:

# Subband processing functions:
- create_filterbank()
- analyze_subbands()
- synthesize_subbands()

Copy

Insert at cursor
python
bit_allocation.py:

# Bit allocation optimization:
- compute_subband_energies()
- optimize_bit_allocation()  # Using Lagrange multiplier
- calculate_perceptual_weights()

Copy

Insert at cursor
python
quantizer.py:

# Quantization implementation:
- quantize_subbands()
- compute_step_sizes()
- apply_quantization()

Copy

Insert at cursor
python
loss_function.py:

# Psychoacoustic loss calculation:
- compute_psychoacoustic_loss()
- calculate_distortion()
- evaluate_bitrate_constraint()

Copy

Insert at cursor
python
main.py:

# Main execution flow:
- load_audio()
- process_frames()
- save_output()'''
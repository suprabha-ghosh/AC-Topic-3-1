import numpy as np
import soundfile as sf
import os
import pandas as pd

# Define input and output paths
INPUT = "data/input_32bit.wav"
QUANTIZED = ["data/quantized_24bit.wav", "data/quantized_16bit.wav"]
RESULTS = "data/quantized_comparison_results.csv"

def compute_snr(original, quantized):
    noise = original - quantized
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else 0
    return snr

def theoretical_snr(bits):
    """Calculate theoretical SNR for given bit depth"""
    return 6.02 * bits + 1.76

def analyze_audio(original_file, quantized_files):
    original, samplerate = sf.read(original_file, dtype='float32')
    results = []
    original_size = os.path.getsize(original_file)
    
    # Add original 32-bit file
    theoretical_32bit_snr = theoretical_snr(32)
    results.append({
        'Audio Format': '32-bit (Original)',
        'File Size (KB)': round(original_size / 1024, 2),
        'SNR (dB)': round(theoretical_32bit_snr, 2),
        'Size Reduction (%)': 0.0
    })
    
    # Add quantized versions
    for file in quantized_files:
        bit_depth = file.split('_')[1].replace('bit.wav', '')
        quantized, _ = sf.read(file, dtype='float32')
        quantized_size = os.path.getsize(file)
        
        # Calculate metrics
        snr = compute_snr(original, quantized)
        size_reduction = (1 - quantized_size / original_size) * 100
        
        results.append({
            'Audio Format': f'{bit_depth}-bit',
            'File Size (KB)': round(quantized_size / 1024, 2),
            'SNR (dB)': round(snr, 2),
            'Size Reduction (%)': round(size_reduction, 2)
        })
    
    df = pd.DataFrame(results)
    
    # Display the table
    print("\nAudio Format Comparison:")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    return df

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Generate and save comparison
df_results = analyze_audio(INPUT, QUANTIZED)
df_results.to_csv(RESULTS, index=False)
print(f"\nResults saved to: {RESULTS}")




'''import numpy as np
import soundfile as sf
import os
import pandas as pd

# Define input and output paths
INPUT = "data/input_32bit.wav"
QUANTIZED = ["data/quantized_24bit.wav", "data/quantized_16bit.wav"]
RESULTS = "data/quantized_comparison_results.csv"

def compute_snr(original, quantized):
    noise = original - quantized
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_psnr(original, quantized):
    max_signal = np.max(np.abs(original))
    mse = np.mean((original - quantized) ** 2)
    psnr = 10 * np.log10(max_signal ** 2 / mse)
    return psnr

def compute_mse(original, quantized):
    return np.mean((original - quantized) ** 2)

def compute_max_error(original, quantized):
    return np.max(np.abs(original - quantized))

def compute_compression_ratio(original_size, quantized_size):
    return original_size / quantized_size

def analyze_audio(original_file, quantized_files):
    original, samplerate = sf.read(original_file, dtype='float32')
    results = []
    original_size = os.path.getsize(original_file)
    
     # Add original 32-bit file metrics
    results.append({
        'Bit Depth': '32',
        'SNR (dB)': 'Reference',
        'PSNR (dB)': 'Reference',
        'MSE': '0',
        'Max Error': '0',
        'File Size (KB)': round(original_size / 1024, 2),
        'Compression Ratio': 1.0,
        'Size Reduction (%)': 0.0,
        'Sample Rate': samplerate
    })
    
    
    for file in quantized_files:
        bit_depth = file.split('_')[1].replace('bit.wav', '')
        quantized, _ = sf.read(file, dtype='float32')
        quantized_size = os.path.getsize(file)
        
        # Compute metrics
        snr = compute_snr(original, quantized)
        psnr = compute_psnr(original, quantized)
        mse = compute_mse(original, quantized)
        max_error = compute_max_error(original, quantized)
        compression_ratio = compute_compression_ratio(original_size, quantized_size)
        size_reduction = (1 - quantized_size / original_size) * 100
        
        # File size in KB
        file_size_kb = quantized_size / 1024
        
        results.append({
            'Bit Depth': bit_depth,
            'SNR (dB)': round(snr, 2),
            'PSNR (dB)': round(psnr, 2),
            'MSE': format(mse, '.2e'),
            'Max Error': format(max_error, '.2e'),
            'File Size (KB)': round(file_size_kb, 2),
            'Compression Ratio': round(compression_ratio, 2),
            'Size Reduction (%)': round(size_reduction, 2),
            'Sample Rate': samplerate
        })
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        'Bit Depth',
        'Sample Rate',
        'File Size (KB)',
        'Compression Ratio',
        'Size Reduction (%)',
        'SNR (dB)',
        'PSNR (dB)',
        'MSE',
        'Max Error'
    ]
    df = df[column_order]
    
    # Add styling to the table output
    print("\nAudio Quantization Analysis Results:")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    return df

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# List of quantized files
quantized_files = QUANTIZED
df_results = analyze_audio(INPUT, quantized_files)

# Save results to CSV
df_results.to_csv(RESULTS, index=False)
print(f"\nResults saved to: {RESULTS}")'''
import numpy as np
import soundfile as sf
import os
import json
import matplotlib.pyplot as plt
import scipy.signal
from scipy import stats
import pandas as pd

def uniform_quantize(data, bit_depth):
    """
    Perform uniform quantization to reduce bit depth.
    :param data: Input signal (array)
    :param bit_depth: Target bit depth (e.g., 24 or 16 bits)
    :return: Quantized signal, step size
    """
    # Calculate the number of possible levels for the target bit depth
    levels = 2**bit_depth
    
    # Scale the input data to the full range [-1, 1]
    max_abs = np.max(np.abs(data))
    if max_abs > 0:
        data_scaled = data / max_abs
    else:
        data_scaled = data
        
    # Calculate step size
    step_size = 2.0 / levels
    
    # Quantize the signal
    quantized = np.round(data_scaled * (levels/2)) * step_size
    
    # Clip to ensure values stay within valid range
    quantized = np.clip(quantized, -1.0, 1.0 - step_size)
    
    return quantized, step_size

def compute_snr(original, quantized):
    """
    Compute Signal-to-Noise Ratio (SNR)
    """
    # Ensure the signals are aligned in amplitude
    scale_factor = np.max(np.abs(original)) / np.max(np.abs(quantized))
    quantized_scaled = quantized * scale_factor
    
    noise = original - quantized_scaled
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr

def analyze_signal(audio_signal, sample_rate):
    """
    Perform comprehensive signal analysis
    """
    analysis = {}
    
    # Time domain analysis
    analysis['peak'] = float(np.max(np.abs(audio_signal)))
    analysis['rms'] = float(np.sqrt(np.mean(audio_signal**2)))
    analysis['crest_factor'] = float(analysis['peak'] / analysis['rms'])
    analysis['dynamic_range'] = float(20 * np.log10(analysis['peak'] / 
                                                   (np.min(np.abs(audio_signal[audio_signal != 0])) + 1e-10)))
    
    # Statistical analysis
    analysis['mean'] = float(np.mean(audio_signal))
    analysis['std'] = float(np.std(audio_signal))
    analysis['skewness'] = float(stats.skew(audio_signal))
    analysis['kurtosis'] = float(stats.kurtosis(audio_signal))
    
    # Frequency domain analysis
    frequencies, power_spectrum = scipy.signal.welch(audio_signal, sample_rate, nperseg=1024)
    analysis['dominant_frequency'] = float(frequencies[np.argmax(power_spectrum)])
    analysis['spectral_centroid'] = float(np.sum(frequencies * power_spectrum) / np.sum(power_spectrum))
    
    return analysis

def plot_analysis(original, quantized, bit_depth, sample_rate, output_dir):
    """
    Generate comprehensive analysis plots
    """
    plt.figure(figsize=(15, 10))
    
    # Time domain comparison - Zoomed in to see quantization levels
    plt.subplot(3, 2, 1)
    start_sample = 1000
    num_samples = 100
    plt.plot(range(start_sample, start_sample + num_samples), 
            original[start_sample:start_sample + num_samples], 
            label="Original", alpha=0.7)
    plt.plot(range(start_sample, start_sample + num_samples), 
            quantized[start_sample:start_sample + num_samples], 
            label=f"Quantized {bit_depth}-bit", linestyle='dashed', alpha=0.7)
    plt.legend()
    plt.title("Time Domain Comparison (Zoomed)")
    plt.grid(True)
    
    # Quantization error
    plt.subplot(3, 2, 2)
    error = original - quantized
    plt.plot(range(start_sample, start_sample + num_samples), 
            error[start_sample:start_sample + num_samples])
    plt.title("Quantization Error (Zoomed)")
    plt.grid(True)
    
    # Frequency spectrum
    plt.subplot(3, 2, 3)
    f_orig, pxx_orig = scipy.signal.welch(original, sample_rate)
    f_quant, pxx_quant = scipy.signal.welch(quantized, sample_rate)
    plt.semilogy(f_orig, pxx_orig, label='Original')
    plt.semilogy(f_quant, pxx_quant, label=f'Quantized {bit_depth}-bit')
    plt.legend()
    plt.title("Power Spectral Density")
    plt.grid(True)
    
    # Error distribution
    plt.subplot(3, 2, 4)
    plt.hist(error, bins=100, density=True)
    plt.title("Error Distribution")
    plt.grid(True)
    
    # Spectrogram comparison
    plt.subplot(3, 2, 5)
    f, t, Sxx = scipy.signal.spectrogram(original, sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10))
    plt.title("Original Spectrogram")
    plt.ylabel('Frequency [Hz]')
    
    plt.subplot(3, 2, 6)
    f, t, Sxx = scipy.signal.spectrogram(quantized, sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10))
    plt.title(f"Quantized Spectrogram ({bit_depth}-bit)")
    plt.ylabel('Frequency [Hz]')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"signal_analysis_{bit_depth}bit.png"))
    plt.close()

def save_results(output_dir, bit_depth, original, quantized, step_size, snr, sample_rate):
    """
    Save results and analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform signal analysis
    orig_analysis = analyze_signal(original, sample_rate)
    quant_analysis = analyze_signal(quantized, sample_rate)
    
    # Create analysis report
    analysis = {
        "bit_depth": int(bit_depth),
        "step_size": float(step_size),
        "snr": float(snr),
        "original": orig_analysis,
        "quantized": quant_analysis
    }
    
    # Save analysis to CSV
    df = pd.DataFrame({
        'Metric': list(orig_analysis.keys()),
        'Original': list(orig_analysis.values()),
        f'Quantized_{bit_depth}bit': list(quant_analysis.values())
    })
    df.to_csv(os.path.join(output_dir, f"analysis_{bit_depth}bit.csv"), index=False)
    
    # Save detailed analysis plots
    plot_analysis(original, quantized, bit_depth, sample_rate, output_dir)
    
    # Save raw signals
    np.save(os.path.join(output_dir, "original_signals.npy"), original)
    np.save(os.path.join(output_dir, "quantized_signals.npy"), quantized)
    
    # Save analysis report
    with open(os.path.join(output_dir, "analysis_report.json"), "w") as f:
        json.dump(analysis, f, indent=4)

def quantize_audio(input_file):
    """
    Load 32-bit WAV, apply quantization (24-bit & 16-bit), save results.
    """
    # Load input audio
    data, sample_rate = sf.read(input_file, dtype='float32')
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    print("Processing audio file...")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(data)/sample_rate:.2f} seconds")
    
    for bit_depth in [24, 16]:
        print(f"\nProcessing {bit_depth}-bit quantization...")
        output_dir = f"quantization_results/{bit_depth}bit"
        
        # Apply uniform quantization
        quantized_data, step_size = uniform_quantize(data, bit_depth)
        
        # Compute SNR
        snr = compute_snr(data, quantized_data)
        print(f"SNR: {snr:.2f} dB")
        print(f"Step size: {step_size:.10f}")
        
        # Save results and analysis
        save_results(output_dir, bit_depth, data, quantized_data, step_size, snr, sample_rate)
        
        # Save quantized audio
        sf.write(os.path.join(output_dir, f"output_{bit_depth}bit.wav"), 
                quantized_data, sample_rate, subtype=f'PCM_{bit_depth}')
    
    print("\nQuantization and analysis completed. Results saved.")

if __name__ == "__main__":
    input_wav = "input_audio/input_audio.wav"
    quantize_audio(input_wav)

import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import json
from scipy.fftpack import dct

class SubbandAnalysis:
    def __init__(self, num_bands=32):
        self.num_bands = num_bands
        self.prototype_filter = self.design_prototype_filter()

    def design_prototype_filter(self):
        """
        Design prototype filter for subband analysis
        """
        # Length of prototype filter (typically 512 for 32 bands)
        M = 512
        
        # Design prototype lowpass filter
        cutoff = 1.0 / (2.0 * self.num_bands)
        prototype = signal.firwin(M, cutoff)
        
        return prototype

    def analyze_subbands(self, audio_data, sample_rate):
        """
        Perform subband analysis of audio signal
        """
        # Initialize subband signals
        subbands = np.zeros((self.num_bands, len(audio_data)))
        
        # Analysis filterbank
        for k in range(self.num_bands):
            # Modulate prototype filter for each subband
            band_filter = self.prototype_filter * np.cos(
                np.pi/(2*self.num_bands) * (2*k + 1) * 
                np.arange(len(self.prototype_filter))
            )
            
            # Filter the signal
            subbands[k] = signal.lfilter(band_filter, 1.0, audio_data)
        
        return subbands

    def compute_subband_energies(self, subbands):
        """
        Compute energy in each subband
        """
        return np.mean(subbands**2, axis=1)

    def plot_subband_spectrum(self, subbands, sample_rate, title, output_file=None):
        """
        Plot spectrum of subband decomposition
        """
        plt.figure(figsize=(12, 6))
        
        # Compute and plot spectrum for each subband
        frequencies = np.fft.rfftfreq(subbands.shape[1], 1/sample_rate)
        for i in range(self.num_bands):
            spectrum = np.abs(np.fft.rfft(subbands[i]))
            plt.semilogy(frequencies, spectrum + 1e-10, alpha=0.5, 
                        label=f'Band {i}' if i % 4 == 0 else "")
        
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'Subband Spectrum - {title}')
        plt.legend()
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def plot_subband_energies(self, energies, title, output_file=None):
        """
        Plot energy distribution across subbands
        """
        plt.figure(figsize=(10, 5))
        plt.bar(range(self.num_bands), 10 * np.log10(energies + 1e-10))
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Energy (dB)')
        plt.title(f'Subband Energy Distribution - {title}')
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

def analyze_audio_files(original_file, bit24_file, bit16_file, output_dir='subband_analysis'):
    """
    Analyze and compare original, 24-bit, and 16-bit audio files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize subband analyzer
    analyzer = SubbandAnalysis(num_bands=32)
    
    # Load audio files
    original, sr_orig = sf.read(original_file)
    audio_24bit, sr_24 = sf.read(bit24_file)
    audio_16bit, sr_16 = sf.read(bit16_file)
    
    # Ensure mono
    if len(original.shape) > 1:
        original = original.mean(axis=1)
    if len(audio_24bit.shape) > 1:
        audio_24bit = audio_24bit.mean(axis=1)
    if len(audio_16bit.shape) > 1:
        audio_16bit = audio_16bit.mean(axis=1)
    
    # Normalize
    original = original / np.max(np.abs(original))
    audio_24bit = audio_24bit / np.max(np.abs(audio_24bit))
    audio_16bit = audio_16bit / np.max(np.abs(audio_16bit))
    
    # Perform subband analysis
    print("Performing subband analysis...")
    
    subbands_orig = analyzer.analyze_subbands(original, sr_orig)
    subbands_24bit = analyzer.analyze_subbands(audio_24bit, sr_24)
    subbands_16bit = analyzer.analyze_subbands(audio_16bit, sr_16)
    
    # Compute subband energies
    energies_orig = analyzer.compute_subband_energies(subbands_orig)
    energies_24bit = analyzer.compute_subband_energies(subbands_24bit)
    energies_16bit = analyzer.compute_subband_energies(subbands_16bit)
    
    # Plot results
    print("Generating plots...")
    
    # Spectrum plots
    analyzer.plot_subband_spectrum(subbands_orig, sr_orig, "Original",
                                 os.path.join(output_dir, 'spectrum_original.png'))
    analyzer.plot_subband_spectrum(subbands_24bit, sr_24, "24-bit",
                                 os.path.join(output_dir, 'spectrum_24bit.png'))
    analyzer.plot_subband_spectrum(subbands_16bit, sr_16, "16-bit",
                                 os.path.join(output_dir, 'spectrum_16bit.png'))
    
    # Energy distribution plots
    analyzer.plot_subband_energies(energies_orig, "Original",
                                 os.path.join(output_dir, 'energy_original.png'))
    analyzer.plot_subband_energies(energies_24bit, "24-bit",
                                 os.path.join(output_dir, 'energy_24bit.png'))
    analyzer.plot_subband_energies(energies_16bit, "16-bit",
                                 os.path.join(output_dir, 'energy_16bit.png'))
    
    # Compute differences
    energy_diff_24 = 10 * np.log10(np.abs(energies_orig - energies_24bit) + 1e-10)
    energy_diff_16 = 10 * np.log10(np.abs(energies_orig - energies_16bit) + 1e-10)
    
    # Save numerical results
    results = {
        'sample_rate': sr_orig,
        'num_subbands': analyzer.num_bands,
        'subband_energies': {
            'original': energies_orig.tolist(),
            '24bit': energies_24bit.tolist(),
            '16bit': energies_16bit.tolist()
        },
        'energy_differences': {
            '24bit_vs_original': energy_diff_24.tolist(),
            '16bit_vs_original': energy_diff_16.tolist()
        }
    }
    
    with open(os.path.join(output_dir, 'subband_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary statistics
    print("\nAnalysis Results:")
    print(f"Number of subbands: {analyzer.num_bands}")
    print("\nEnergy Differences (dB):")
    print(f"24-bit vs Original - Max: {np.max(energy_diff_24):.2f}, Mean: {np.mean(energy_diff_24):.2f}")
    print(f"16-bit vs Original - Max: {np.max(energy_diff_16):.2f}, Mean: {np.mean(energy_diff_16):.2f}")
    
    return results

if __name__ == "__main__":
    # Specify input files
    original_file = "input_audio/input_audio.wav"
    bit24_file = "quantization_results/24bit/output_24bit.wav"
    bit16_file = "quantization_results/16bit/output_16bit.wav"
    
    try:
        results = analyze_audio_files(original_file, bit24_file, bit16_file)
        print("\nSubband analysis completed successfully!")
    except FileNotFoundError as e:
        print(f"Error: Could not find audio file - {str(e)}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

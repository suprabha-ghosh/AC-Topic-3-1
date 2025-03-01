import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import windows
import matplotlib.pyplot as plt
import os
import json
import soundfile as sf

class PsychoacousticModel:
    def __init__(self, sample_rate=44100):
        """
        Initialize the psychoacoustic model
        """
        self.sample_rate = sample_rate
        
        # Bark scale critical band edges (Hz)
        self.critical_bands = [
            20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
            2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
        ]
        
        # Absolute threshold of hearing (dB SPL)
        self.threshold_quiet = [
            96, 51, 32, 24, 20, 16, 13, 12, 11, 10, 9, 9, 9,
            9, 9, 10, 11, 12, 14, 17, 21, 25, 30, 35, 40
        ]

    def compute_critical_bands(self, fft_freqs):
        """
        Compute critical band indices for given FFT frequencies
        """
        bark_bands = []
        for freq in fft_freqs:
            if freq < 20:  # Below first critical band
                bark_bands.append(0)
            elif freq > 15500:  # Above last critical band
                bark_bands.append(len(self.critical_bands) - 2)
            else:
                # Find appropriate critical band
                for i in range(len(self.critical_bands) - 1):
                    if self.critical_bands[i] <= freq < self.critical_bands[i + 1]:
                        bark_bands.append(i)
                        break
        return np.array(bark_bands)

    def spreading_function(self, bark_diff):
        """
        Compute spreading function in Bark domain
        """
        if bark_diff >= -3 and bark_diff <= 8:
            spread = (15.81 + 7.5 * (bark_diff + 0.474) - 
                     17.5 * np.sqrt(1 + (bark_diff + 0.474)**2))
        else:
            spread = -1000  # Effectively zero masking
        return spread

    def compute_masking_threshold(self, signal_data, window_size=2048):
        """
        Compute masking threshold for signal
        """
        # Create window
        window = windows.hann(window_size)
        
        # Calculate number of frames and prepare arrays
        hop_size = window_size // 2  # 50% overlap
        num_frames = (len(signal_data) - window_size) // hop_size + 1
        freqs = np.fft.rfftfreq(window_size, 1/self.sample_rate)
        
        # Initialize arrays for averaging
        avg_power_spectrum = np.zeros(len(freqs))
        avg_masking_threshold = np.zeros(len(freqs))
        
        # Process each frame
        for i in range(num_frames):
            # Extract frame
            start = i * hop_size
            frame = signal_data[start:start + window_size]
            
            # Apply window and compute FFT
            windowed_frame = frame * window
            spectrum = np.fft.rfft(windowed_frame)
            power_spectrum = 20 * np.log10(np.abs(spectrum) + 1e-10)
            
            # Add to average
            avg_power_spectrum += power_spectrum
            
            # Get critical band indices
            bark_bands = self.compute_critical_bands(freqs)
            
            # Initialize frame masking threshold
            frame_masking_threshold = np.zeros_like(freqs)
            
            # Compute threshold in quiet
            quiet_interp = interp1d(self.critical_bands, self.threshold_quiet, 
                                  kind='cubic', fill_value='extrapolate')
            threshold_quiet = quiet_interp(freqs)
            
            # For each critical band
            for band in range(len(self.critical_bands) - 1):
                band_mask = (bark_bands == band)
                if not np.any(band_mask):
                    continue
                    
                # Find maximum power in band
                band_power = power_spectrum[band_mask]
                max_power = np.max(band_power)
                
                # Compute spreading function
                for j, freq in enumerate(freqs):
                    target_band = bark_bands[j]
                    bark_diff = target_band - band
                    spread = self.spreading_function(bark_diff)
                    
                    # Add masking contribution
                    frame_masking_threshold[j] = 10 * np.log10(
                        10**(frame_masking_threshold[j]/10) + 
                        10**((max_power + spread)/10)
                    )
            
            # Consider threshold in quiet
            frame_masking_threshold = np.maximum(frame_masking_threshold, threshold_quiet)
            avg_masking_threshold += frame_masking_threshold
        
        # Compute averages
        if num_frames > 0:
            avg_power_spectrum /= num_frames
            avg_masking_threshold /= num_frames
        
        return avg_masking_threshold, freqs

    def analyze_signal(self, signal_data, window_size=2048):
        """
        Perform complete psychoacoustic analysis
        """
        # Compute masking threshold
        masking_threshold, freqs = self.compute_masking_threshold(signal_data, window_size)
        
        # Compute power spectrum
        window = windows.hann(window_size)
        hop_size = window_size // 2
        num_frames = (len(signal_data) - window_size) // hop_size + 1
        
        # Initialize average power spectrum
        avg_power_spectrum = np.zeros(window_size//2 + 1)
        
        # Process each frame
        for i in range(num_frames):
            start = i * hop_size
            frame = signal_data[start:start + window_size]
            windowed_frame = frame * window
            spectrum = np.fft.rfft(windowed_frame)
            power_spectrum = 20 * np.log10(np.abs(spectrum) + 1e-10)
            avg_power_spectrum += power_spectrum
        
        # Compute average
        if num_frames > 0:
            avg_power_spectrum /= num_frames
        
        # Get critical bands
        bark_bands = self.compute_critical_bands(freqs)
        
        # Compute band energies
        band_energies = {}
        for band in range(len(self.critical_bands) - 1):
            band_mask = (bark_bands == band)
            if np.any(band_mask):
                band_energies[f"band_{band}"] = float(np.mean(avg_power_spectrum[band_mask]))
        
        return {
            'frequencies': freqs,
            'power_spectrum': avg_power_spectrum,
            'masking_threshold': masking_threshold,
            'critical_bands': bark_bands,
            'band_energies': band_energies
        }

    def plot_analysis(self, masking_thresholds, output_file=None):
        """
        Plot psychoacoustic analysis results
        """
        plt.figure(figsize=(12, 8))
        
        # Plot power spectrum and masking threshold
        plt.semilogx(masking_thresholds['frequencies'], 
                     masking_thresholds['power_spectrum'], 
                     label='Signal')
        plt.semilogx(masking_thresholds['frequencies'], 
                     masking_thresholds['masking_threshold'], 
                     label='Masking Threshold')
        
        # Plot threshold in quiet
        quiet_interp = interp1d(self.critical_bands, self.threshold_quiet, 
                               kind='cubic', fill_value='extrapolate')
        threshold_quiet = quiet_interp(masking_thresholds['frequencies'])
        plt.semilogx(masking_thresholds['frequencies'], threshold_quiet, 
                     '--', label='Threshold in Quiet')
        
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.title('Psychoacoustic Analysis')
        plt.legend()
        
        if output_file:
            plt.savefig(output_file)
        plt.close()

def test_psychoacoustic_model(input_audio_file):
    """
    Test function to demonstrate usage of PsychoacousticModel with real audio file
    """
    try:
        # Load the audio file
        print(f"Loading audio file: {input_audio_file}")
        signal_data, sample_rate = sf.read(input_audio_file)
        
        # Convert to mono if stereo
        if len(signal_data.shape) > 1:
            signal_data = signal_data.mean(axis=1)
        
        # Normalize the signal
        signal_data = signal_data / np.max(np.abs(signal_data))
        
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(signal_data)/sample_rate:.2f} seconds")
        
        # Create instance of PsychoacousticModel
        model = PsychoacousticModel(sample_rate=sample_rate)
        
        # Create output directory
        output_dir = 'psychoacoustic_analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze signal
        print("Computing psychoacoustic analysis...")
        analysis = model.analyze_signal(signal_data)
        
        # Plot and save results
        print("Generating plots...")
        model.plot_analysis(analysis, os.path.join(output_dir, 'masking_threshold_analysis.png'))
        
        # Save numerical results
        results = {
            'sample_rate': sample_rate,
            'duration': len(signal_data)/sample_rate,
            'band_energies': analysis['band_energies']
        }
        
        with open(os.path.join(output_dir, 'masking_thresholds.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Analysis completed. Results saved in '{output_dir}' directory.")
        
        # Print key metrics
        print("\nKey Analysis Metrics:")
        print(f"Number of critical bands analyzed: {len(analysis['band_energies'])}")
        print(f"Frequency range analyzed: {analysis['frequencies'][0]:.1f} Hz to {analysis['frequencies'][-1]:.1f} Hz")
        print(f"Maximum masking threshold: {np.max(analysis['masking_threshold']):.2f} dB")
        print(f"Minimum masking threshold: {np.min(analysis['masking_threshold']):.2f} dB")
        
        return analysis
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Specify your input audio file
    input_wav = "input_audio/input_audio.wav"
    
    try:
        analysis = test_psychoacoustic_model(input_wav)
        print("\nPsychoacoustic analysis successful!")
    except FileNotFoundError:
        print(f"Error: Could not find audio file at {input_wav}")
        print("Please ensure the audio file exists and the path is correct.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

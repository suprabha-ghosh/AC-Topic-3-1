import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import windows
import matplotlib.pyplot as plt
import os

class SubbandAnalysis:
    def __init__(self, num_subbands, frame_size, sample_rate):
        """
        Initialize subband analysis
        
        Args:
            num_subbands (int): Number of subbands (64, 128, 512, 1024, or 2048)
            frame_size (int): Size of each frame
            sample_rate (int): Audio sample rate
        """
        self.valid_subbands = [64, 128, 512, 1024, 2048]
        if num_subbands not in self.valid_subbands:
            raise ValueError(f"num_subbands must be one of {self.valid_subbands}")
            
        self.num_subbands = num_subbands
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.freq_bins = np.fft.fftfreq(frame_size, 1/sample_rate)
        self.filters = self._create_filterbank()

    def _create_filterbank(self):
        """
        Create subband analysis filterbank with mel-scaled frequency bands
        
        Returns:
            np.ndarray: Filter bank coefficients
        """
        filters = np.zeros((self.num_subbands, self.frame_size))
        
        # Calculate mel-scaled frequency edges
        mel_max = 2595 * np.log10(1 + (self.sample_rate/2)/700)
        mel_points = np.linspace(0, mel_max, self.num_subbands + 1)
        freq_edges = 700 * (10**(mel_points/2595) - 1)
        
        # Normalize frequencies
        norm_freq_edges = freq_edges / (self.sample_rate/2)
        freq = np.linspace(0, 1, self.frame_size)
        
        for i in range(self.num_subbands):
            freq_low = norm_freq_edges[i]
            freq_high = norm_freq_edges[i+1]
            
            # Create filter response with transitions
            transition_width = 0.1 * (freq_high - freq_low)
            response = np.zeros(self.frame_size)
            
            # Define regions
            lower_transition = np.where(
                (freq >= freq_low - transition_width) & (freq < freq_low)
            )[0]
            passband = np.where(
                (freq >= freq_low) & (freq < freq_high)
            )[0]
            upper_transition = np.where(
                (freq >= freq_high) & (freq < freq_high + transition_width)
            )[0]
            
            # Set filter response
            response[passband] = 1.0
            if len(lower_transition) > 0:
                response[lower_transition] = 0.5 * (1 + np.cos(
                    np.pi * (freq_low - freq[lower_transition]) / transition_width
                ))
            if len(upper_transition) > 0:
                response[upper_transition] = 0.5 * (1 + np.cos(
                    np.pi * (freq[upper_transition] - freq_high) / transition_width
                ))
            
            response *= windows.hann(self.frame_size)
            filters[i] = response
            
        # Normalize filters
        filters /= np.max(np.abs(filters), axis=1)[:, np.newaxis]
        
        return filters

    def analyze_frame(self, frame):
        """
        Analyze a single frame and prepare data for bit allocation
        
        Args:
            frame (np.ndarray): Input audio frame
            
        Returns:
            dict: Analysis results including subband signals and energies
        """
        if len(frame) != self.frame_size:
            raise ValueError(f"Frame size must be {self.frame_size}")
            
        # Normalize frame
        frame = frame / (np.max(np.abs(frame)) + 1e-10)
        
        # Compute spectrum
        spectrum = fft(frame)
        subband_signals = np.zeros((self.num_subbands, self.frame_size), dtype=complex)
        subband_energies = np.zeros(self.num_subbands)
        
        # Analyze each subband
        for i in range(self.num_subbands):
            # Apply filter
            subband_signals[i] = spectrum * self.filters[i]
            
            # Calculate energy
            subband_energies[i] = np.sum(np.abs(subband_signals[i])**2)
        
        # Calculate additional statistics for bit allocation
        energy_variance = np.var(subband_energies)
        peak_frequencies = np.array([
            np.abs(self.freq_bins[np.argmax(np.abs(signal))])
            for signal in subband_signals
        ])
        
        return {
            'subband_signals': subband_signals,
            'subband_energies': subband_energies,
            'energy_variance': energy_variance,
            'peak_frequencies': peak_frequencies,
            'frame': frame
        }

    def synthesize_frame(self, subband_signals, bits_per_subband=None):
        """
        Reconstruct signal from subbands with optional quantization
        
        Args:
            subband_signals (np.ndarray): Subband signals
            bits_per_subband (np.ndarray, optional): Bits allocated per subband
            
        Returns:
            np.ndarray: Reconstructed signal
        """
        if bits_per_subband is not None:
            # Apply quantization based on bit allocation
            quantized_signals = np.zeros_like(subband_signals)
            for i in range(self.num_subbands):
                if bits_per_subband[i] > 0:
                    # Simple uniform quantization
                    max_val = np.max(np.abs(subband_signals[i]))
                    levels = 2**bits_per_subband[i]
                    step = (2 * max_val) / levels
                    quantized = np.round(subband_signals[i] / step) * step
                    quantized_signals[i] = quantized
            subband_signals = quantized_signals
            
        reconstructed = np.sum(subband_signals, axis=0)
        return np.real(ifft(reconstructed))

    def plot_analysis(self, analysis_results, bits_per_subband=None, output_dir="subband_analysis"):
        """
        Plot analysis results
        
        Args:
            analysis_results (dict): Results from analyze_frame
            bits_per_subband (np.ndarray, optional): Bit allocation per subband
            output_dir (str): Output directory for plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Reconstruct signal
        reconstructed = self.synthesize_frame(
            analysis_results['subband_signals'],
            bits_per_subband
        )
        
        plt.figure(figsize=(15, 10))
        
        # 1. Original vs Reconstructed Signal
        plt.subplot(2, 2, 1)
        plt.plot(analysis_results['frame'], label='Original', alpha=0.7)
        plt.plot(reconstructed, label='Reconstructed', alpha=0.7)
        plt.grid(True)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('Signal Reconstruction')
        plt.legend()
        
        # 2. Filterbank Response
        plt.subplot(2, 2, 2)
        for i in range(min(5, self.num_subbands)):  # Plot first 5 filters
            plt.plot(self.freq_bins[:len(self.freq_bins)//2], 
                    self.filters[i][:len(self.freq_bins)//2],
                    alpha=0.7,
                    label=f'Band {i+1}')
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Filterbank Frequency Response')
        plt.legend()
        
        # 3. Subband Energies
        plt.subplot(2, 2, 3)
        energies_db = 10 * np.log10(analysis_results['subband_energies'] + 1e-10)
        plt.plot(energies_db, alpha=0.7)
        plt.grid(True)
        plt.xlabel('Subband')
        plt.ylabel('Energy (dB)')
        plt.title('Subband Energy Distribution')
        
        # 4. Bit Allocation (if provided)
        plt.subplot(2, 2, 4)
        if bits_per_subband is not None:
            plt.plot(bits_per_subband, alpha=0.7)
            plt.grid(True)
            plt.xlabel('Subband')
            plt.ylabel('Allocated Bits')
            plt.title('Bit Allocation')
        else:
            plt.plot(analysis_results['peak_frequencies'], alpha=0.7)
            plt.grid(True)
            plt.xlabel('Subband')
            plt.ylabel('Frequency (Hz)')
            plt.title('Peak Frequencies')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/subband_analysis_{self.num_subbands}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    try:
        # Example usage
        frame_size = 2048
        sample_rate = 44100
        
        # Create test signal
        t = np.linspace(0, 1, frame_size)
        test_signal = (np.sin(2 * np.pi * 440 * t) + 
                      np.sin(2 * np.pi * 1000 * t) +
                      np.sin(2 * np.pi * 4000 * t))
        
        # Test different subband configurations
        for num_subbands in [64, 128, 512, 1024, 2048]:
            print(f"\nAnalyzing with {num_subbands} subbands...")
            
            # Initialize analyzer
            analyzer = SubbandAnalysis(num_subbands, frame_size, sample_rate)
            
            # Analyze frame
            results = analyzer.analyze_frame(test_signal)
            
            # Plot results
            analyzer.plot_analysis(results)
            
            print(f"Analysis completed for {num_subbands} subbands")
            print(f"Energy variance: {results['energy_variance']:.2e}")
            print(f"Number of non-zero subbands: "
                  f"{np.sum(results['subband_energies'] > 1e-10)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

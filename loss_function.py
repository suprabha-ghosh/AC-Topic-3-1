import numpy as np
from scipy.fftpack import fft
from scipy.signal import windows

class PsychoacousticLoss:
    def __init__(self, sample_rate=44100):
        """
        Initialize PsychoacousticLoss with parameters
        
        Args:
            sample_rate (int): Sampling rate of the audio signal
        """
        self.sample_rate = sample_rate
        self.window = windows.hann  # Window function for analysis
        self.bark_scale = self._init_bark_scale()
        
    def _init_bark_scale(self):
        """
        Initialize Bark scale parameters for critical band analysis
        """
        return {
            'alpha': 13,  # Bark scale parameter
            'beta': 3.5,  # Bark scale parameter
            'freq_min': 20,    # Minimum frequency (Hz)
            'freq_max': 20000  # Maximum frequency (Hz)
        }
    
    def _frequency_to_bark(self, freq):
        """
        Convert frequency to Bark scale
        
        Args:
            freq (np.ndarray): Frequencies in Hz
            
        Returns:
            np.ndarray: Frequencies in Bark scale
        """
        return (self.bark_scale['alpha'] * np.arctan(0.00076 * freq) + 
                self.bark_scale['beta'] * np.arctan((freq / 7500.0) ** 2))
    
    def _compute_spreading_function(self, bark_diff):
        """
        Compute spreading function for masking threshold calculation
        
        Args:
            bark_diff (np.ndarray): Difference in Bark scale
            
        Returns:
            np.ndarray: Spreading function values
        """
        return 15.81 + 7.5 * (bark_diff + 0.474) - 17.5 * np.sqrt(1.0 + (bark_diff + 0.474) ** 2)
    
    def _compute_masking_threshold(self, spectrum, freq):
        """
        Compute masking threshold based on spectrum
        
        Args:
            spectrum (np.ndarray): Power spectrum of the signal
            freq (np.ndarray): Frequency array
            
        Returns:
            np.ndarray: Masking threshold
        """
        # Convert to power in dB
        power_db = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-10)
        
        # Convert frequencies to Bark scale
        bark = self._frequency_to_bark(freq)
        
        # Initialize masking threshold
        masking = np.zeros_like(power_db)
        
        # Compute spreading function for each frequency
        for i in range(len(freq)):
            if freq[i] >= self.bark_scale['freq_min'] and freq[i] <= self.bark_scale['freq_max']:
                bark_diff = bark - bark[i]
                spread = self._compute_spreading_function(bark_diff)
                masking = np.maximum(masking, power_db[i] + spread)
        
        return masking
    
    def compute_loss(self, original, quantized, masking_threshold=None):
        """
        Compute psychoacoustic loss between original and quantized signals
        
        Args:
            original (np.ndarray): Original audio signal
            quantized (np.ndarray): Quantized audio signal
            masking_threshold (np.ndarray, optional): Pre-computed masking threshold
            
        Returns:
            dict: Dictionary containing various loss metrics
        """
        # Apply window
        window = self.window(len(original))
        original_windowed = original * window
        quantized_windowed = quantized * window
        
        # Convert to frequency domain
        orig_spectrum = fft(original_windowed)
        quant_spectrum = fft(quantized_windowed)
        
        # Compute frequencies
        freq = np.fft.fftfreq(len(original), 1/self.sample_rate)
        pos_freq_mask = freq >= 0  # Consider only positive frequencies
        
        # Compute or use provided masking threshold
        if masking_threshold is None:
            masking_threshold = self._compute_masking_threshold(orig_spectrum, freq)
        
        # Compute error in dB
        error_magnitude = np.abs(orig_spectrum - quant_spectrum)
        error_db = 20 * np.log10(error_magnitude + 1e-10)
        
        # Weight error by masking threshold
        weighted_error = np.maximum(0, error_db - masking_threshold)
        
        # Compute various metrics
        metrics = {
            'weighted_error': np.mean(weighted_error[pos_freq_mask]),
            'peak_error': np.max(weighted_error[pos_freq_mask]),
            'error_above_threshold': np.sum(weighted_error > 0) / len(weighted_error),
            'mse': np.mean(error_magnitude ** 2),
            'snr': 10 * np.log10(np.mean(np.abs(orig_spectrum) ** 2) / 
                                (np.mean(error_magnitude ** 2) + 1e-10))
        }
        
        return metrics

def main():
    """
    Test function for PsychoacousticLoss
    """
    # Test parameters
    sample_rate = 44100
    duration = 0.1  # seconds
    
    # Generate test signals
    t = np.arange(int(duration * sample_rate)) / sample_rate
    
    # Original signal (440 Hz tone)
    original = np.sin(2 * np.pi * 440 * t)
    
    # Simulated quantized signal (with some noise)
    quantized = original + 0.01 * np.random.randn(len(original))
    
    # Initialize loss calculator
    loss_calculator = PsychoacousticLoss(sample_rate=sample_rate)
    
    # Compute loss
    loss_metrics = loss_calculator.compute_loss(original, quantized)
    
    # Print results
    print("\nPsychoacoustic Loss Metrics:")
    for metric, value in loss_metrics.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()

import numpy as np

class BitAllocation:
    def __init__(self, target_bitrate):
        """
        Initialize bit allocation with target bitrate and bit-depth specific parameters
        
        Args:
            target_bitrate (int): Target bit depth (32, 24, or 16)
        """
        self.target_bitrate = target_bitrate
        self.lambda_min = 1e-10
        self.lambda_max = 100
        
        # Set quantization parameters based on target bit depth
        if target_bitrate == 32:
            self.max_bits_per_sample = 32
            self.noise_floor = -192  # dB for 32-bit audio
            self.dynamic_range = 192
        elif target_bitrate == 24:
            self.max_bits_per_sample = 24
            self.noise_floor = -144  # dB for 24-bit audio
            self.dynamic_range = 144
        elif target_bitrate == 16:
            self.max_bits_per_sample = 16
            self.noise_floor = -96   # dB for 16-bit audio
            self.dynamic_range = 96
        else:
            raise ValueError("Target bitrate must be 32, 24, or 16")
        
        self.min_bits_per_band = 2  # Minimum bits to allocate to any band
        
    def compute_subband_energies(self, subband_signals):
        """
        Compute energy in each subband
        
        Args:
            subband_signals (np.ndarray): Subband decomposed signals
            
        Returns:
            np.ndarray: Energy values for each subband
        """
        return np.mean(np.abs(subband_signals) ** 2, axis=1)
    
    def compute_perceptual_entropy(self, subband_energies, masking_threshold):
        """
        Compute perceptual entropy for each subband
        
        Args:
            subband_energies (np.ndarray): Energy in each subband
            masking_threshold (np.ndarray): Masking threshold for each subband
            
        Returns:
            np.ndarray: Perceptual entropy values
        """
        signal_to_mask_ratio = 10 * np.log10(subband_energies / (masking_threshold + 1e-10))
        pe = np.maximum(0, signal_to_mask_ratio) / 6.02  # 6.02 dB per bit
        return pe
    
    def optimize_allocation(self, subband_energies, masking_threshold, frame_size):
        """
        Optimize bit allocation using Lagrange multiplier with bit depth constraints
        
        Args:
            subband_energies (np.ndarray): Energy in each subband
            masking_threshold (np.ndarray): Masking threshold for each subband
            frame_size (int): Size of the audio frame
            
        Returns:
            np.ndarray: Optimized bit allocation for each subband
        """
        # Calculate target total bits based on bit depth
        target_bits = self.target_bitrate * frame_size
        
        # Compute perceptual entropy
        pe = self.compute_perceptual_entropy(subband_energies, masking_threshold)
        
        def compute_bits(lambda_val):
            """
            Compute bit allocation for a given lambda value
            """
            # Perceptual weight based on masking threshold and dynamic range
            perceptual_weight = 1 / (masking_threshold + 1e-10)
            perceptual_weight *= (self.dynamic_range / 96.0)  # Scale based on bit depth
            
            # Basic bit allocation formula
            bits = 0.5 * np.log2(subband_energies * perceptual_weight / lambda_val)
            
            # Apply perceptual entropy weighting
            bits *= (pe / np.max(pe + 1e-10))
            
            # Scale bits based on target bit depth
            bits *= (self.target_bitrate / 16.0)  # Scale relative to 16-bit
            
            # Apply bit depth constraints
            bits = np.minimum(bits, self.max_bits_per_sample)
            bits = np.maximum(bits, 0)
            
            # Ensure minimum bits for active bands
            active_bands = bits > 0
            bits[active_bands] = np.maximum(bits[active_bands], self.min_bits_per_band)
            
            return bits
        
        # Binary search for optimal lambda
        lambda_min = self.lambda_min
        lambda_max = self.lambda_max
        best_bits = None
        min_error = float('inf')
        
        for _ in range(100):
            lambda_mid = (lambda_min + lambda_max) / 2
            bits = compute_bits(lambda_mid)
            total_bits = np.sum(bits)
            
            error = abs(total_bits - target_bits)
            
            if error < min_error:
                min_error = error
                best_bits = bits
            
            if abs(total_bits - target_bits) < 1e-6:
                break
            elif total_bits > target_bits:
                lambda_min = lambda_mid
            else:
                lambda_max = lambda_mid
        
        return self.refine_bit_allocation(best_bits, subband_energies, masking_threshold)

    def refine_bit_allocation(self, initial_bits, subband_energies, masking_threshold):
        """
        Refine bit allocation to better match psychoacoustic criteria
        
        Args:
            initial_bits (np.ndarray): Initial bit allocation
            subband_energies (np.ndarray): Energy in each subband
            masking_threshold (np.ndarray): Masking threshold for each subband
            
        Returns:
            np.ndarray: Refined bit allocation
        """
        bits = initial_bits.copy()
        
        # Calculate SNR for each band
        snr = 10 * np.log10(subband_energies / (masking_threshold + 1e-10))
        
        # Scale SNR based on bit depth
        snr *= (self.target_bitrate / 16.0)
        
        # Normalize SNR
        snr_normalized = (snr - np.min(snr)) / (np.max(snr) - np.min(snr) + 1e-10)
        
        # Adjust bits based on SNR
        active_bands = bits > 0
        if np.any(active_bands):
            # Redistribute bits from less perceptually important bands
            total_adjustment = np.sum(snr_normalized[active_bands] * 0.1)
            bits[active_bands] += snr_normalized[active_bands] * total_adjustment
            
            # Scale adjustment based on bit depth
            bits *= (self.target_bitrate / 16.0)
            
            # Ensure we don't exceed maximum bits per sample
            bits = np.minimum(bits, self.max_bits_per_sample)
            
            # Round to nearest integer
            bits = np.round(bits)
        
        return bits

    def verify_allocation(self, bit_allocation, frame_size):
        """
        Verify that bit allocation meets constraints
        
        Args:
            bit_allocation (np.ndarray): Bit allocation array
            frame_size (int): Size of the audio frame
            
        Returns:
            bool: True if allocation is valid, False otherwise
        """
        total_bits = np.sum(bit_allocation)
        target_bits = self.target_bitrate * frame_size
        
        # Allow margin based on bit depth
        margin = 1.1 + (self.target_bitrate - 16) * 0.01
        
        if total_bits > target_bits * margin:
            return False
        
        if np.any(bit_allocation > self.max_bits_per_sample):
            return False
        
        if np.any(np.logical_and(bit_allocation > 0, 
                                bit_allocation < self.min_bits_per_band)):
            return False
        
        return True

    def calculate_allocation_statistics(self, bit_allocation):
        """
        Calculate statistics about the bit allocation
        
        Args:
            bit_allocation (np.ndarray): Bit allocation array
            
        Returns:
            dict: Statistics about the bit allocation
        """
        stats = {
            'total_bits': np.sum(bit_allocation),
            'max_bits': np.max(bit_allocation),
            'min_bits': np.min(bit_allocation[bit_allocation > 0]),
            'mean_bits': np.mean(bit_allocation[bit_allocation > 0]),
            'active_bands': np.sum(bit_allocation > 0),
            'bit_depth': self.target_bitrate,
            'dynamic_range': self.dynamic_range,
            'theoretical_snr': 6.02 * self.target_bitrate + 1.76  # Theoretical SNR
        }
        return stats

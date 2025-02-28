# quantizer.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import logging
import os
import json
import matplotlib.pyplot as plt

@dataclass
class QuantizationInfo:
    step_sizes: np.ndarray
    scale_factors: np.ndarray
    snr_db: np.ndarray
    noise_energy: np.ndarray
    bit_allocation: np.ndarray

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Quantizer:
    def __init__(self, target_bitrate: int):
        """Initialize quantizer with target bit rate"""
        self.target_bitrate = target_bitrate
        self.logger = logging.getLogger(__name__)
        self._validate_bitrate()
        
    def _validate_bitrate(self):
        """Validate the target bitrate"""
        valid_bitrates = [16, 24, 32]
        if self.target_bitrate not in valid_bitrates:
            raise ValueError(f"Target bitrate must be one of {valid_bitrates}")
        
    def compute_step_sizes(self, 
                         subband_signals: np.ndarray, 
                         bit_allocation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute quantization step sizes for each subband"""
        num_subbands = len(bit_allocation)
        step_sizes = np.zeros(num_subbands)
        scale_factors = np.zeros(num_subbands)
        
        # Compute maximum amplitude and scale factors
        max_amplitudes = np.max(np.abs(subband_signals), axis=1)
        active_bands = (bit_allocation > 0) & (max_amplitudes > 0)
        
        # Vectorized computation for active bands
        scale_factors[active_bands] = max_amplitudes[active_bands]
        num_levels = 2 ** bit_allocation[active_bands]
        
        # Compute step sizes based on bit depth
        if self.target_bitrate == 32:
            step_sizes[active_bands] = 2.0 / (num_levels * 1.5)
        elif self.target_bitrate == 24:
            step_sizes[active_bands] = 2.0 / num_levels
        else:  # 16-bit
            step_sizes[active_bands] = 2.0 / (num_levels * 0.9)
            
        return step_sizes, scale_factors

    def apply_quantization(self, 
                         subband_signals: np.ndarray,
                         bit_allocation: np.ndarray,
                         step_sizes: np.ndarray,
                         scale_factors: np.ndarray) -> np.ndarray:
        """Apply quantization to normalized subband signals"""
        # Create mask for active bands
        active_bands = (bit_allocation > 0) & (scale_factors > 0)
        
        # Initialize output array
        quantized_signals = np.zeros_like(subband_signals)
        
        if not np.any(active_bands):
            self.logger.warning("No active bands found for quantization")
            return quantized_signals
        
        # Process active bands
        for i in np.where(active_bands)[0]:
            # Normalize and quantize
            normalized = subband_signals[i] / scale_factors[i]
            quantized = np.round(normalized / step_sizes[i])
            
            # Clip to valid range
            max_value = (2 ** (bit_allocation[i] - 1)) - 1
            quantized = np.clip(quantized, -max_value, max_value)
            
            # Scale back
            quantized_signals[i] = quantized * step_sizes[i] * scale_factors[i]
        
        return quantized_signals

    def quantize_subbands(self, 
                         subband_signals: np.ndarray,
                         bit_allocation: np.ndarray) -> Tuple[np.ndarray, QuantizationInfo]:
        """Main quantization function"""
        try:
            # Compute step sizes and scale factors
            step_sizes, scale_factors = self.compute_step_sizes(subband_signals, bit_allocation)
            
            # Apply quantization
            quantized_signals = self.apply_quantization(
                subband_signals, bit_allocation, step_sizes, scale_factors
            )
            
            # Calculate noise and SNR
            noise = subband_signals - quantized_signals
            noise_energy = np.mean(noise ** 2, axis=1)
            signal_energy = np.mean(subband_signals ** 2, axis=1)
            snr = 10 * np.log10(signal_energy / (noise_energy + 1e-10))
            
            # Create quantization info
            quant_info = QuantizationInfo(
                step_sizes=step_sizes,
                scale_factors=scale_factors,
                snr_db=snr,
                noise_energy=noise_energy,
                bit_allocation=bit_allocation
            )
            
            return quantized_signals, quant_info
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {str(e)}")
            raise

    def analyze_performance(self, 
                          original: np.ndarray, 
                          quantized: np.ndarray, 
                          quant_info: QuantizationInfo) -> Dict:
        """Analyze quantization performance"""
        error = original - quantized
        mse = np.mean(error ** 2)
        max_error = np.max(np.abs(error))
        
        subband_mse = np.mean(error ** 2, axis=1)
        subband_snr = 10 * np.log10(
            np.mean(original ** 2, axis=1) / 
            (np.mean(error ** 2, axis=1) + 1e-10)
        )
        
        return {
            'overall': {
                'mse': float(mse),
                'max_error': float(max_error),
                'avg_snr': float(np.mean(subband_snr))
            },
            'per_subband': {
                'mse': subband_mse.tolist(),
                'snr': subband_snr.tolist(),
                'step_sizes': quant_info.step_sizes.tolist(),
                'scale_factors': quant_info.scale_factors.tolist()
            }
        }

    def save_results(self, 
                    original_signals: np.ndarray,
                    quantized_signals: np.ndarray,
                    quant_info: QuantizationInfo,
                    output_dir: str = "quantization_results"):
        """Save quantization results and analysis"""
        # Create output directory
        bit_depth_dir = os.path.join(output_dir, f"{self.target_bitrate}bit")
        os.makedirs(bit_depth_dir, exist_ok=True)

        # 1. Save quantization parameters
        params = {
            'target_bitrate': self.target_bitrate,
            'num_subbands': len(quant_info.bit_allocation),
            'bit_allocation': quant_info.bit_allocation,
            'step_sizes': quant_info.step_sizes,
            'scale_factors': quant_info.scale_factors
        }
        
        with open(os.path.join(bit_depth_dir, 'quantization_params.json'), 'w') as f:
            json.dump(params, f, cls=NumpyEncoder, indent=4)

        # 2. Save performance analysis
        analysis = self.analyze_performance(original_signals, quantized_signals, quant_info)
        
        with open(os.path.join(bit_depth_dir, 'performance_analysis.json'), 'w') as f:
            json.dump(analysis, f, cls=NumpyEncoder, indent=4)

        # 3. Generate and save plots
        self._save_analysis_plots(original_signals, quantized_signals, quant_info, bit_depth_dir)

        # 4. Save numerical data
        np.save(os.path.join(bit_depth_dir, 'original_signals.npy'), original_signals)
        np.save(os.path.join(bit_depth_dir, 'quantized_signals.npy'), quantized_signals)

        return bit_depth_dir

    def _save_analysis_plots(self, 
                           original_signals: np.ndarray,
                           quantized_signals: np.ndarray,
                           quant_info: QuantizationInfo,
                           output_dir: str):
        """Generate and save analysis plots"""
        
        # 1. Signal Comparison Plot
        plt.figure(figsize=(12, 6))
        for i in range(min(3, len(original_signals))):  # Plot first 3 subbands
            plt.subplot(3, 1, i+1)
            plt.plot(original_signals[i, :100], label='Original', alpha=0.7)
            plt.plot(quantized_signals[i, :100], label='Quantized', alpha=0.7)
            plt.title(f'Subband {i} Signal Comparison')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'signal_comparison.png'))
        plt.close()

        # 2. SNR and Bit Allocation
        plt.figure(figsize=(10, 6))
        plt.plot(quant_info.snr_db, label='SNR (dB)')
        plt.plot(quant_info.bit_allocation, label='Bit Allocation')
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('dB / Bits')
        plt.title('SNR and Bit Allocation')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'snr_bits.png'))
        plt.close()

        # 3. Quantization Error
        error = original_signals - quantized_signals
        plt.figure(figsize=(10, 6))
        plt.plot(np.mean(error**2, axis=1))
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Mean Squared Error')
        plt.title('Quantization Error by Subband')
        plt.savefig(os.path.join(output_dir, 'quantization_error.png'))
        plt.close()

def test_quantizer():
    """Unit test function for both 16-bit and 24-bit quantization"""
    # Test setup
    frame_size = 1024
    num_subbands = 32
    
    # Create test data
    subband_signals = np.random.randn(num_subbands, frame_size)
    bit_allocation = np.random.randint(0, 8, num_subbands)
    
    # Test both 16-bit and 24-bit quantization
    for target_bitrate in [16, 24]:
        print(f"\nTesting {target_bitrate}-bit quantization:")
        
        try:
            # Initialize quantizer
            quantizer = Quantizer(target_bitrate)
            
            # Test quantization
            quantized_signals, quant_info = quantizer.quantize_subbands(
                subband_signals, bit_allocation
            )
            
            # Save results
            output_dir = quantizer.save_results(
                subband_signals,
                quantized_signals,
                quant_info
            )
            
            print(f"Results saved to: {output_dir}")
            
            # Basic assertions
            assert quantized_signals.shape == subband_signals.shape, "Shape mismatch"
            assert np.all(np.isfinite(quantized_signals)), "Non-finite values found"
            assert len(quant_info.step_sizes) == num_subbands, "Step sizes length mismatch"
            
            # Compare SNR between bit depths
            mean_snr = np.mean(quant_info.snr_db)
            print(f"Mean SNR: {mean_snr:.2f} dB")
            
            # Theoretical SNR improvement (6.02 dB per bit)
            if target_bitrate == 24:
                theoretical_improvement = (24 - 16) * 6.02
                print(f"Theoretical SNR improvement over 16-bit: {theoretical_improvement:.2f} dB")
            
            print(f"{target_bitrate}-bit quantization tests passed successfully!")
            
        except Exception as e:
            print(f"Error during {target_bitrate}-bit testing: {str(e)}")
            return False
    
    return True

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_quantizer()

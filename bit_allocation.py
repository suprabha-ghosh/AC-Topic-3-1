import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import fsolve

class SubbandBitAllocation:
    def __init__(self, num_bands, target_bitrate):
        """
        Initialize subband bit allocation analyzer
        
        Args:
            num_bands: Number of subbands (64, 128, or 512)
            target_bitrate: Target bitrate in bits per sample
        """
        self.num_bands = num_bands
        self.target_bitrate = target_bitrate
        self.prototype_filter = self.design_prototype_filter()

    def design_prototype_filter(self):
        """Design prototype filter for subband analysis"""
        M = self.num_bands * 16  # Filter length proportional to number of subbands
        cutoff = 1.0 / (2.0 * self.num_bands)
        return signal.firwin(M, cutoff)

    def analyze_subbands(self, audio_data):
        """Decompose signal into subbands"""
        subbands = np.zeros((self.num_bands, len(audio_data)))
        
        for k in range(self.num_bands):
            band_filter = self.prototype_filter * np.cos(
                np.pi/(2*self.num_bands) * (2*k + 1) * 
                np.arange(len(self.prototype_filter))
            )
            subbands[k] = signal.lfilter(band_filter, 1.0, audio_data)
        
        return subbands

    def compute_subband_variances(self, subbands):
        """Compute variance in each subband"""
        return np.var(subbands, axis=1)

    def lagrange_bit_allocation(self, variances):
        """
        Implement Lagrange multiplier method for optimal bit allocation
        """
        def bit_constraint(lambda_val):
            bits = 0.5 * np.log2(variances / lambda_val)
            bits = np.maximum(bits, 0)  # No negative bits
            return np.sum(bits) - self.target_bitrate * self.num_bands

        # Find optimal Lagrange multiplier
        lambda_init = np.mean(variances) / (2 ** (2 * self.target_bitrate))
        lambda_opt = fsolve(bit_constraint, lambda_init)[0]

        # Compute final bit allocation
        bits = 0.5 * np.log2(variances / lambda_opt)
        return np.maximum(bits, 0)

    def analyze_and_allocate(self, audio_data, sample_rate, label):
        """
        Perform complete analysis and bit allocation for one audio version
        """
        # Perform subband analysis
        subbands = self.analyze_subbands(audio_data)
        
        # Compute subband variances
        variances = self.compute_subband_variances(subbands)
        
        # Perform bit allocation
        allocated_bits = self.lagrange_bit_allocation(variances)
        
        # Calculate SNR per subband
        snr_per_band = 6.02 * allocated_bits + 1.76  # Theoretical SNR formula
        
        return {
            'label': label,
            'variances': variances,
            'allocated_bits': allocated_bits,
            'snr_per_band': snr_per_band,
            'avg_bits': np.mean(allocated_bits),
            'max_bits': np.max(allocated_bits),
            'min_bits': np.min(allocated_bits),
            'std_bits': np.std(allocated_bits)
        }

    def plot_comparison(self, results, output_dir):
        """
        Plot comparison of different audio versions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Subband Variances
        plt.figure(figsize=(12, 6))
        for result in results:
            plt.semilogy(range(self.num_bands), result['variances'], 
                        label=result['label'])
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Variance')
        plt.title(f'Subband Signal Variances ({self.num_bands} bands)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'variances_{self.num_bands}bands.png'))
        plt.close()
        
        # Plot 2: Bit Allocation
        plt.figure(figsize=(12, 6))
        for result in results:
            plt.plot(range(self.num_bands), result['allocated_bits'], 
                    label=result['label'])
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Allocated Bits')
        plt.title(f'Bit Allocation across Subbands ({self.num_bands} bands)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'bit_allocation_{self.num_bands}bands.png'))
        plt.close()
        
        # Plot 3: SNR per Subband
        plt.figure(figsize=(12, 6))
        for result in results:
            plt.plot(range(self.num_bands), result['snr_per_band'], 
                    label=result['label'])
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('SNR (dB)')
        plt.title(f'Theoretical SNR per Subband ({self.num_bands} bands)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'snr_{self.num_bands}bands.png'))
        plt.close()

def analyze_multiple_configurations(input_file, bit24_file, bit16_file, target_bitrate=16):
    """
    Analyze audio with different subband configurations and bit depths
    """
    # Subband configurations to test
    configurations = [64, 128, 512]
    
    # Load audio files
    input_audio, sr_input = sf.read(input_file)
    audio_24bit, sr_24 = sf.read(bit24_file)
    audio_16bit, sr_16 = sf.read(bit16_file)
    
    # Ensure mono
    if len(input_audio.shape) > 1:
        input_audio = input_audio.mean(axis=1)
    if len(audio_24bit.shape) > 1:
        audio_24bit = audio_24bit.mean(axis=1)
    if len(audio_16bit.shape) > 1:
        audio_16bit = audio_16bit.mean(axis=1)
    
    # Normalize
    input_audio = input_audio / np.max(np.abs(input_audio))
    audio_24bit = audio_24bit / np.max(np.abs(audio_24bit))
    audio_16bit = audio_16bit / np.max(np.abs(audio_16bit))
    
    results = {}
    
    for num_bands in configurations:
        print(f"\nAnalyzing {num_bands} subbands configuration...")
        
        # Create analyzer
        analyzer = SubbandBitAllocation(num_bands, target_bitrate)
        
        # Create output directory
        output_dir = f'bit_allocation_{num_bands}'
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Analyze all versions
            results_original = analyzer.analyze_and_allocate(
                input_audio, sr_input, "Original")
            results_24bit = analyzer.analyze_and_allocate(
                audio_24bit, sr_24, "24-bit")
            results_16bit = analyzer.analyze_and_allocate(
                audio_16bit, sr_16, "16-bit")
            
            # Convert numpy arrays to lists for JSON serialization
            for results_dict in [results_original, results_24bit, results_16bit]:
                results_dict['variances'] = results_dict['variances'].tolist()
                results_dict['allocated_bits'] = results_dict['allocated_bits'].tolist()
                results_dict['snr_per_band'] = results_dict['snr_per_band'].tolist()
                results_dict['avg_bits'] = float(results_dict['avg_bits'])
                results_dict['max_bits'] = float(results_dict['max_bits'])
                results_dict['min_bits'] = float(results_dict['min_bits'])
                results_dict['std_bits'] = float(results_dict['std_bits'])
            
            # Plot comparisons
            analyzer.plot_comparison(
                [results_original, results_24bit, results_16bit],
                output_dir
            )
            
            # Store results
            results[num_bands] = {
                'original': results_original,
                '24bit': results_24bit,
                '16bit': results_16bit
            }
            
            # Save numerical results
            with open(os.path.join(output_dir, 'allocation_results.json'), 'w') as f:
                json.dump(results[num_bands], f, indent=4)
            
            # Print summary statistics
            print(f"\nConfiguration: {num_bands} subbands")
            for version in [results_original, results_24bit, results_16bit]:
                print(f"\n{version['label']}:")
                print(f"Average bits per subband: {version['avg_bits']:.2f}")
                print(f"Max bits allocated: {version['max_bits']:.2f}")
                print(f"Min bits allocated: {version['min_bits']:.2f}")
                print(f"Standard deviation: {version['std_bits']:.2f}")
            
        except Exception as e:
            print(f"Error analyzing {num_bands} subbands: {str(e)}")
            raise  # Add this to see the full error traceback
    
    return results

def compare_configurations(results):
    """
    Compare results across different subband configurations
    """
    plt.figure(figsize=(15, 8))
    
    # Plot bit allocation for each configuration and version
    for num_bands, result in results.items():
        for version in ['original', '24bit', '16bit']:
            allocated_bits = np.array(result[version]['allocated_bits'])  # Convert back to numpy array
            plt.plot(np.linspace(0, 1, len(allocated_bits)), 
                    allocated_bits, 
                    label=f'{num_bands} bands - {version}')
    
    plt.grid(True)
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Allocated Bits')
    plt.title('Bit Allocation Comparison Across Configurations and Bit Depths')
    plt.legend()
    plt.savefig('subband_comparison_all.png')
    plt.close()


if __name__ == "__main__":
    # Specify input files
    input_file = "input_audio/input_audio.wav"
    bit24_file = "quantization_results/24bit/output_24bit.wav"
    bit16_file = "quantization_results/16bit/output_16bit.wav"
    
    try:
        # Analyze all configurations
        results = analyze_multiple_configurations(input_file, bit24_file, bit16_file)
        
        # Compare configurations
        compare_configurations(results)
        
        print("\nAnalysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find audio file - {str(e)}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

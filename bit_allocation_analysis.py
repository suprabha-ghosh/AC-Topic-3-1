import numpy as np
import json
import csv
import os
import matplotlib.pyplot as plt
from bit_allocation import BitAllocation

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

class BitAllocationAnalysis(BitAllocation):
    def save_results(self, bit_allocation, subband_energies, masking_threshold, 
                    frame_size, output_dir="bit_allocation_results"):
        """
        Save bit allocation results to files
        
        Args:
            bit_allocation (np.ndarray): Optimized bit allocation
            subband_energies (np.ndarray): Energy in each subband
            masking_threshold (np.ndarray): Masking threshold for each subband
            frame_size (int): Size of the audio frame
            output_dir (str): Output directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectory for this bit depth
        bit_depth_dir = os.path.join(output_dir, f"{self.target_bitrate}bit")
        os.makedirs(bit_depth_dir, exist_ok=True)

        # 1. Save numerical results to CSV
        numerical_results = {
            'subband_index': np.arange(len(bit_allocation)),
            'bit_allocation': bit_allocation,
            'subband_energy_db': 10 * np.log10(subband_energies + 1e-10),
            'masking_threshold_db': 10 * np.log10(masking_threshold + 1e-10)
        }
        
        csv_path = os.path.join(bit_depth_dir, 'numerical_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=numerical_results.keys())
            writer.writeheader()
            for i in range(len(bit_allocation)):
                row = {k: float(numerical_results[k][i]) for k in numerical_results.keys()}
                writer.writerow(row)

        # 2. Save statistics to JSON
        stats = self.calculate_allocation_statistics(bit_allocation)
        stats['frame_size'] = int(frame_size)
        stats['verification'] = bool(self.verify_allocation(bit_allocation, frame_size))
        
        json_path = os.path.join(bit_depth_dir, 'statistics.json')
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=4, cls=NumpyEncoder)

        # 3. Generate and save plots
        self._save_analysis_plots(bit_allocation, subband_energies, 
                                masking_threshold, bit_depth_dir)

        return bit_depth_dir

    def _save_analysis_plots(self, bit_allocation, subband_energies, 
                           masking_threshold, output_dir):
        """
        Generate and save analysis plots
        
        Args:
            bit_allocation (np.ndarray): Bit allocation array
            subband_energies (np.ndarray): Energy in each subband
            masking_threshold (np.ndarray): Masking threshold for each subband
            output_dir (str): Output directory
        """
        # 1. Bit Allocation Distribution
        plt.figure(figsize=(10, 6))
        plt.plot(bit_allocation, 'b-', label='Bit Allocation')
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Allocated Bits')
        plt.title(f'Bit Allocation Distribution ({self.target_bitrate}-bit)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'bit_allocation.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Energy and Masking Threshold
        plt.figure(figsize=(10, 6))
        plt.plot(10 * np.log10(subband_energies + 1e-10), 'g-', 
                label='Subband Energy')
        plt.plot(10 * np.log10(masking_threshold + 1e-10), 'r--', 
                label='Masking Threshold')
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Magnitude (dB)')
        plt.title(f'Energy and Masking Threshold ({self.target_bitrate}-bit)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'energy_masking.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Bits vs Energy
        plt.figure(figsize=(10, 6))
        plt.scatter(10 * np.log10(subband_energies + 1e-10), 
                   bit_allocation, alpha=0.6)
        plt.grid(True)
        plt.xlabel('Subband Energy (dB)')
        plt.ylabel('Allocated Bits')
        plt.title(f'Bit Allocation vs Energy ({self.target_bitrate}-bit)')
        plt.savefig(os.path.join(output_dir, 'bits_vs_energy.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 4. SNR Analysis
        plt.figure(figsize=(10, 6))
        snr = 10 * np.log10(subband_energies / (masking_threshold + 1e-10))
        plt.plot(snr, 'r-', label='SNR')
        plt.plot(bit_allocation, 'b--', label='Bit Allocation')
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('SNR (dB) / Bits')
        plt.title(f'SNR and Bit Allocation ({self.target_bitrate}-bit)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'snr_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Cumulative Bit Distribution
        plt.figure(figsize=(10, 6))
        cumulative_bits = np.cumsum(bit_allocation)
        plt.plot(cumulative_bits, 'g-')
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Cumulative Bits')
        plt.title(f'Cumulative Bit Distribution ({self.target_bitrate}-bit)')
        plt.savefig(os.path.join(output_dir, 'cumulative_bits.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def save_comparative_analysis(results_32, results_24, results_16, 
                            output_dir="bit_allocation_results"):
    """
    Save comparative analysis of different bit depths
    
    Args:
        results_32 (dict): Results from 32-bit allocation
        results_24 (dict): Results from 24-bit allocation
        results_16 (dict): Results from 16-bit allocation
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Comparative Bit Allocation Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_32['bit_allocation'], label='32-bit', alpha=0.7)
    plt.plot(results_24['bit_allocation'], label='24-bit', alpha=0.7)
    plt.plot(results_16['bit_allocation'], label='16-bit', alpha=0.7)
    plt.grid(True)
    plt.xlabel('Subband Index')
    plt.ylabel('Allocated Bits')
    plt.title('Comparative Bit Allocation')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'comparative_allocation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Save comparative statistics
    comparative_stats = {
        '32bit': results_32['statistics'],
        '24bit': results_24['statistics'],
        '16bit': results_16['statistics']
    }
    
    with open(os.path.join(output_dir, 'comparative_statistics.json'), 'w') as f:
        json.dump(comparative_stats, f, indent=4, cls=NumpyEncoder)

def main():
    """
    Test function with output saving
    """
    # Test parameters
    frame_size = 1024
    num_subbands = 32
    
    # Create test data
    subband_energies = np.random.exponential(1, num_subbands)
    masking_threshold = np.random.exponential(0.1, num_subbands)
    
    results = {}
    
    # Test for all bit depths
    for target_bitrate in [32, 24, 16]:
        print(f"\nProcessing {target_bitrate}-bit allocation:")
        
        # Initialize analyzer
        analyzer = BitAllocationAnalysis(target_bitrate)
        
        # Optimize allocation
        bit_allocation = analyzer.optimize_allocation(
            subband_energies, 
            masking_threshold, 
            frame_size
        )
        
        # Save results
        output_dir = analyzer.save_results(
            bit_allocation,
            subband_energies,
            masking_threshold,
            frame_size
        )
        
        # Store results for comparative analysis
        results[target_bitrate] = {
            'bit_allocation': bit_allocation,
            'statistics': analyzer.calculate_allocation_statistics(bit_allocation)
        }
        
        print(f"Results saved to: {output_dir}")
    
    # Generate comparative analysis
    save_comparative_analysis(
        results[32],
        results[24],
        results[16]
    )
    
    print("\nAnalysis complete. All results saved.")

if __name__ == "__main__":
    main()

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

class FinalOptimizer:
    def __init__(self):
        self.results_dir = 'final_optimization_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def load_all_results(self, num_bands):
        """
        Load all results for a specific band configuration
        """
        try:
            # Load masking thresholds
            with open('psychoacoustic_analysis/masking_thresholds.json', 'r') as f:
                masking_data = json.load(f)

            # Load Lagrange optimization results
            lagrange_path = f'lagrange_optimization_analysis/analysis_{num_bands}bands/optimization_results.json'
            with open(lagrange_path, 'r') as f:
                lagrange_results = json.load(f)

            # Load bit allocation results
            bit_alloc_path = f'bit_allocation_{num_bands}/allocation_results.json'
            with open(bit_alloc_path, 'r') as f:
                bit_allocation = json.load(f)

            return masking_data, lagrange_results, bit_allocation

        except Exception as e:
            print(f"Error loading results for {num_bands} bands: {str(e)}")
            return None, None, None

    def compute_final_allocation(self, original_bits, optimized_bits, masking_threshold):
        """
        Compute final bit allocation considering both original and optimized allocations
        """
        try:
            original_bits = np.array(original_bits)
            optimized_bits = np.array(optimized_bits)
            masking_threshold = np.array(masking_threshold)

            # Normalize masking threshold as weights
            weights = 1.0 / (np.maximum(masking_threshold, -120) + 1e-10)
            weights = weights / np.sum(weights)

            # Compute weighted final allocation
            final_bits = np.round(0.7 * optimized_bits + 0.3 * original_bits)

            # Maintain total bit constraint
            target_total = np.sum(original_bits)
            current_total = np.sum(final_bits)

            if current_total != target_total:
                diff = int(target_total - current_total)
                if diff > 0:
                    indices = np.argsort(weights)[-diff:]
                    final_bits[indices] += 1
                else:
                    indices = np.argsort(weights)[:abs(diff)]
                    final_bits[indices] -= 1

            return final_bits.astype(int)

        except Exception as e:
            print(f"Error in final allocation computation: {str(e)}")
            return None

    def compute_snr(self, signal_power, bit_allocation):
        """
        Compute SNR for each subband
        """
        try:
            bit_allocation = np.array(bit_allocation)
            quantization_noise_power = 1 / (2 ** (2 * bit_allocation))  # Approximation

            snr_values = 10 * np.log10(signal_power / quantization_noise_power)
            return snr_values

        except Exception as e:
            print(f"Error computing SNR: {str(e)}")
            return None

    def analyze_allocation(self, original_bits, optimized_bits, final_bits):
        """
        Analyze different bit allocations
        """
        return {
            'original': {
                'mean': float(np.mean(original_bits)),
                'std': float(np.std(original_bits)),
                'max': int(np.max(original_bits)),
                'min': int(np.min(original_bits))
            },
            'optimized': {
                'mean': float(np.mean(optimized_bits)),
                'std': float(np.std(optimized_bits)),
                'max': int(np.max(optimized_bits)),
                'min': int(np.min(optimized_bits))
            },
            'final': {
                'mean': float(np.mean(final_bits)),
                'std': float(np.std(final_bits)),
                'max': int(np.max(final_bits)),
                'min': int(np.min(final_bits))
            }
        }

    def create_comparison_plots(self, num_bands, bit_depth, original_bits, optimized_bits, final_bits, masking_threshold, snr_values):
        """
        Create comparison plots for different bit allocations and SNR
        """
        plt.figure(figsize=(15, 10))

        # Bit allocation comparison
        plt.subplot(211)
        plt.plot(original_bits, 'b-', label=f'Original {bit_depth}', alpha=0.7)
        plt.plot(optimized_bits, 'r--', label='Lagrange Optimized', alpha=0.7)
        plt.plot(final_bits, 'g-', label='Final Allocation', linewidth=2)
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Allocated Bits')
        plt.title(f'Bit Allocation Comparison - {num_bands} bands, {bit_depth}')
        plt.legend()

        # SNR plot
        plt.subplot(212)
        plt.plot(snr_values, 'm-', label='SNR (dB)', linewidth=2)
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('SNR (dB)')
        plt.title('Signal-to-Noise Ratio (SNR) per Subband')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'comparison_snr_{num_bands}bands_{bit_depth}.png'))
        plt.close()

    def process_configuration(self, num_bands):
        """
        Process a specific band configuration
        """
        print(f"\nProcessing {num_bands} bands configuration...")

        masking_data, lagrange_results, bit_allocation = self.load_all_results(num_bands)
        if any(x is None for x in [masking_data, lagrange_results, bit_allocation]):
            return None

        results = {}
        bit_depths = ['original', '24bit', '16bit']

        # Get masking threshold and handle interpolation if needed
        available_bands = len(masking_data['band_energies'])
        available_energies = []
        
        # Collect available energies
        for i in range(available_bands):
            key = f'band_{i}'
            if key in masking_data['band_energies']:
                available_energies.append(masking_data['band_energies'][key])
        
        # Convert to numpy array and interpolate if needed
        available_energies = np.array(available_energies)
        if num_bands != available_bands:
            print(f"Interpolating masking thresholds from {available_bands} to {num_bands} bands")
            x_original = np.linspace(0, 1, available_bands)
            x_target = np.linspace(0, 1, num_bands)
            masking_threshold = np.interp(x_target, x_original, available_energies)
        else:
            masking_threshold = available_energies

        for bit_depth in bit_depths:
            print(f"Processing {bit_depth}...")

            try:
                # Get bit allocations
                original_bits = np.array(bit_allocation[bit_depth]['allocated_bits'])
                optimized_bits = np.array(lagrange_results[bit_depth]['optimized']['bits'])
                
                # Compute final allocation
                final_bits = self.compute_final_allocation(
                    original_bits, optimized_bits, masking_threshold)
                
                if final_bits is None:
                    continue

                # Compute SNR
                signal_power = np.array(bit_allocation[bit_depth]['variances'])
                snr_values = self.compute_snr(signal_power, final_bits)

                # Analyze results
                analysis = self.analyze_allocation(original_bits, optimized_bits, final_bits)

                # Create plots
                self.create_comparison_plots(
                    num_bands, bit_depth, original_bits, optimized_bits, 
                    final_bits, masking_threshold, snr_values)

                results[bit_depth] = {
                    'analysis': analysis,
                    'final_allocation': final_bits.tolist(),
                    'total_bits': int(np.sum(final_bits)),
                    'snr_values': snr_values.tolist()
                }

            except Exception as e:
                print(f"Error processing {bit_depth}: {str(e)}")
                continue

        return results

    def run_optimization(self, configurations=[64, 128, 512]):
        """
        Run final optimization for all configurations
        """
        final_results = {}
        summary_data = []  # For CSV generation

        for num_bands in configurations:
            results = self.process_configuration(num_bands)
            if results:
                final_results[num_bands] = results
                
                # Collect data for CSV
                for bit_depth, data in results.items():
                    analysis = data['analysis']
                    snr_values = np.array(data['snr_values'])
                    
                    summary_data.append({
                        'Num_Bands': num_bands,
                        'Bit_Depth': bit_depth,
                        'Total_Bits': data['total_bits'],
                        'Original_Mean_Bits': analysis['original']['mean'],
                        'Original_Max_Bits': analysis['original']['max'],
                        'Original_Min_Bits': analysis['original']['min'],
                        'Original_Std_Bits': analysis['original']['std'],
                        'Optimized_Mean_Bits': analysis['optimized']['mean'],
                        'Optimized_Max_Bits': analysis['optimized']['max'],
                        'Optimized_Min_Bits': analysis['optimized']['min'],
                        'Optimized_Std_Bits': analysis['optimized']['std'],
                        'Final_Mean_Bits': analysis['final']['mean'],
                        'Final_Max_Bits': analysis['final']['max'],
                        'Final_Min_Bits': analysis['final']['min'],
                        'Final_Std_Bits': analysis['final']['std'],
                        'Mean_SNR': float(np.mean(snr_values)),
                        'Min_SNR': float(np.min(snr_values)),
                        'Max_SNR': float(np.max(snr_values))
                    })

        # Save JSON results
        with open(os.path.join(self.results_dir, 'final_optimization_results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)

        # Save CSV summary
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(self.results_dir, 'optimization_summary.csv')
            df.to_csv(csv_path, index=False)
            
            # Print summary
            print("\nOptimization Summary:")
            print(df.to_string(index=False))
            print(f"\nSummary saved to: {csv_path}")

        print("\nFinal optimization completed successfully!")
        return final_results


if __name__ == "__main__":
    optimizer = FinalOptimizer()
    optimizer.run_optimization()

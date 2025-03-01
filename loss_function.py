import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

class PsychoacousticLossAnalyzer:
    def __init__(self, num_bands):
        self.num_bands = num_bands
        
    def compute_spectral_loss(self, original, processed, masking_threshold):
        """
        Compute spectral domain loss considering masking threshold
        """
        # Compute error
        error = np.abs(original - processed)
        
        # Apply masking threshold
        masked_error = np.maximum(0, error - masking_threshold)
        
        metrics = {
            'mse': float(np.mean(error ** 2)),
            'masked_mse': float(np.mean(masked_error ** 2)),
            'max_error': float(np.max(error)),
            'above_threshold': float(np.sum(masked_error > 0) / len(masked_error))
        }
        
        return metrics, error, masked_error

def load_masking_threshold(num_bands):
    """
    Load and process masking threshold data
    """
    try:
        with open('psychoacoustic_analysis/masking_thresholds.json', 'r') as f:
            masking_data = json.load(f)
            
        # Extract band energies as array
        band_energies = []
        available_bands = len(masking_data['band_energies'])
        
        # If we have fewer bands in masking data than requested
        if available_bands < num_bands:
            print(f"Warning: Masking threshold data has {available_bands} bands, but {num_bands} were requested.")
            print("Padding with zeros for remaining bands.")
            
            # Get available bands
            for i in range(available_bands):
                key = f'band_{i}'
                band_energies.append(masking_data['band_energies'][key])
            
            # Pad with zeros for remaining bands
            band_energies.extend([0] * (num_bands - available_bands))
        else:
            # Get requested number of bands
            for i in range(num_bands):
                key = f'band_{i}'
                if key in masking_data['band_energies']:
                    band_energies.append(masking_data['band_energies'][key])
                else:
                    raise KeyError(f"Missing band {i} in masking threshold data")
                
        return np.array(band_energies)
    
    except FileNotFoundError:
        raise FileNotFoundError("Masking thresholds file not found")
    except KeyError as e:
        print(f"Warning: {str(e)}")
        print("Using default masking threshold values")
        # Return default masking threshold (can be adjusted based on your needs)
        return np.zeros(num_bands) - 60  # -60 dB default threshold

def load_bit_allocation_results(num_bands):
    """
    Load bit allocation results
    """
    folder_path = f'bit_allocation_{num_bands}'
    file_path = os.path.join(folder_path, 'allocation_results.json')
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Bit allocation results not found for {num_bands} bands")

def analyze_loss(configurations=[64, 128, 512]):
    """
    Analyze loss for all configurations
    """
    main_output_dir = 'loss_function_analysis'
    os.makedirs(main_output_dir, exist_ok=True)
    
    results = {}
    
    for num_bands in configurations:
        print(f"\nAnalyzing loss for {num_bands} subbands configuration...")
        
        output_dir = os.path.join(main_output_dir, f'analysis_{num_bands}bands')
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load data
            bit_alloc = load_bit_allocation_results(num_bands)
            
            # Get masking threshold (with error handling)
            try:
                masking_threshold = load_masking_threshold(num_bands)
            except Exception as e:
                print(f"Warning: Error loading masking threshold: {str(e)}")
                print("Using default masking threshold values")
                masking_threshold = np.zeros(num_bands) - 60  # -60 dB default threshold
            
            # Initialize analyzer
            analyzer = PsychoacousticLossAnalyzer(num_bands)
            
            # Process results
            results[num_bands] = {
                'original': {},
                '24bit': {},
                '16bit': {}
            }
            
            # Get original variances
            original_variances = np.array(bit_alloc['original']['variances'])
            
            # Ensure masking threshold length matches variance length
            if len(masking_threshold) != len(original_variances):
                print(f"Warning: Masking threshold length ({len(masking_threshold)}) " +
                      f"doesn't match variance length ({len(original_variances)})")
                # Pad or trim masking threshold to match
                if len(masking_threshold) < len(original_variances):
                    masking_threshold = np.pad(
                        masking_threshold,
                        (0, len(original_variances) - len(masking_threshold)),
                        'constant',
                        constant_values=-60
                    )
                else:
                    masking_threshold = masking_threshold[:len(original_variances)]
            
            # Process each version
            for version in ['24bit', '16bit']:
                version_variances = np.array(bit_alloc[version]['variances'])
                version_bits = np.array(bit_alloc[version]['allocated_bits'])
                
                # Compute losses
                metrics, error, masked_error = analyzer.compute_spectral_loss(
                    original_variances,
                    version_variances,
                    masking_threshold
                )
                
                # Store results
                results[num_bands][version] = {
                    'metrics': metrics,
                    'error': error.tolist(),
                    'masked_error': masked_error.tolist(),
                    'allocated_bits': version_bits.tolist(),
                    'snr': 10 * np.log10(np.mean(original_variances) / 
                                       np.mean((original_variances - version_variances) ** 2))
                }
            
            # Save results
            with open(os.path.join(output_dir, 'loss_metrics.json'), 'w') as f:
                json.dump(results[num_bands], f, indent=4)
            
            # Create plots
            create_comparison_plots(
                num_bands,
                bit_alloc,
                masking_threshold,
                results[num_bands],
                output_dir
            )
            
            # Create summary
            create_summary_dataframe(num_bands, results[num_bands], output_dir)
            
        except Exception as e:
            print(f"Error analyzing {num_bands} subbands: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall results
    if results:
        with open(os.path.join(main_output_dir, 'all_configurations_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    return results, main_output_dir

def create_comparison_plots(num_bands, bit_alloc, masking_threshold, results, output_dir):
    """
    Create comparison plots
    """
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Variance Comparison
    plt.subplot(411)
    plt.semilogy(bit_alloc['original']['variances'], label='Original')
    for version in ['24bit', '16bit']:
        plt.semilogy(bit_alloc[version]['variances'], label=version)
    plt.grid(True)
    plt.xlabel('Subband Index')
    plt.ylabel('Variance (log scale)')
    plt.title(f'Subband Variance Comparison ({num_bands} bands)')
    plt.legend()
    
    # Plot 2: Bit Allocation
    plt.subplot(412)
    for version in ['24bit', '16bit']:
        plt.plot(results[version]['allocated_bits'], label=version)
    plt.grid(True)
    plt.xlabel('Subband Index')
    plt.ylabel('Allocated Bits')
    plt.title('Bit Allocation Comparison')
    plt.legend()
    
    # Plot 3: Error Distribution
    plt.subplot(413)
    for version in ['24bit', '16bit']:
        plt.semilogy(results[version]['error'], label=f'{version} Error')
    plt.grid(True)
    plt.xlabel('Subband Index')
    plt.ylabel('Error Magnitude (log scale)')
    plt.title('Error Distribution')
    plt.legend()
    
    # Plot 4: Masking Threshold vs Error
    plt.subplot(414)
    plt.plot(masking_threshold, 'k--', label='Masking Threshold')
    for version in ['24bit', '16bit']:
        plt.plot(results[version]['masked_error'], label=f'{version} Masked Error')
    plt.grid(True)
    plt.xlabel('Subband Index')
    plt.ylabel('Magnitude (dB)')
    plt.title('Masking Threshold vs Masked Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_analysis_plots.png'))
    plt.close()

def create_summary_dataframe(num_bands, results, output_dir):
    """
    Create summary metrics
    """
    summary_data = []
    for version in ['24bit', '16bit']:
        metrics = results[version]['metrics']
        summary_data.append({
            'Version': version,
            'Num Bands': num_bands,
            'SNR (dB)': results[version]['snr'],
            'MSE': metrics['mse'],
            'Masked MSE': metrics['masked_mse'],
            'Max Error': metrics['max_error'],
            'Samples Above Threshold (%)': metrics['above_threshold'] * 100
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    print(f"\nResults for {num_bands} subbands:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    try:
        # Run analysis
        results, output_dir = analyze_loss()
        
        if results:
            print("\nAnalysis completed successfully!")
            
            # Create final summary
            final_summary = []
            for num_bands in results:
                for version in ['24bit', '16bit']:
                    metrics = results[num_bands][version]['metrics']
                    final_summary.append({
                        'Num Bands': num_bands,
                        'Version': version,
                        'SNR (dB)': results[num_bands][version]['snr'],
                        'MSE': metrics['mse'],
                        'Masked MSE': metrics['masked_mse'],
                        'Samples Above Threshold (%)': metrics['above_threshold'] * 100
                    })
            
            df_final = pd.DataFrame(final_summary)
            df_final.to_csv(os.path.join(output_dir, 'final_summary.csv'), index=False)
            print("\nFinal Summary:")
            print(df_final.to_string(index=False))
        else:
            print("No results were generated.")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from loss_function import PsychoacousticLossAnalyzer

class LagrangeOptimizer:
    def __init__(self, num_bands):
        self.num_bands = num_bands

    def optimize_bit_allocation(self, variances, masking_thresholds, total_bits, psychoacoustic_loss):
        try:
            # Convert inputs to numpy arrays and handle NaN values
            variances = np.array(variances, dtype=float)
            masking_thresholds = np.array(masking_thresholds, dtype=float)
            psychoacoustic_loss = np.array(psychoacoustic_loss, dtype=float)
            
            # Replace NaN and infinite values
            epsilon = 1e-10
            variances = np.nan_to_num(variances, nan=epsilon, posinf=1e10, neginf=epsilon)
            masking_thresholds = np.nan_to_num(masking_thresholds, nan=-60, posinf=0, neginf=-120)
            psychoacoustic_loss = np.nan_to_num(psychoacoustic_loss, nan=0, posinf=1, neginf=0)
            
            # Ensure positive variances
            variances = np.maximum(variances, epsilon)
            
            # Initialize lambda search
            lambda_min = epsilon
            lambda_max = np.max(variances) * 10
            max_iterations = 100
            tolerance = 1e-6
            
            # Water-filling algorithm
            best_bits = np.full(len(variances), total_bits / len(variances))
            best_total = float('inf')
            
            for _ in range(max_iterations):
                lambda_mid = (lambda_min + lambda_max) / 2
                
                # Compute weights
                weights = 1.0 / (np.maximum(masking_thresholds, -120) + epsilon)
                weights *= (1.0 + psychoacoustic_loss)
                weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
                weights /= np.sum(weights)
                
                # Compute bits
                bits = 0.5 * np.log2(weights * variances / lambda_mid)
                bits = np.nan_to_num(bits, nan=0.0, posinf=16.0, neginf=0.0)
                bits = np.maximum(bits, 0)
                
                current_total = np.sum(bits)
                
                if abs(current_total - total_bits) < abs(best_total - total_bits):
                    best_bits = bits
                    best_total = current_total
                
                if abs(current_total - total_bits) < tolerance:
                    break
                elif current_total > total_bits:
                    lambda_min = lambda_mid
                else:
                    lambda_max = lambda_mid
            
            # Round to integers
            rounded_bits = np.floor(best_bits).astype(int)
            remainder = int(total_bits - np.sum(rounded_bits))
            
            if remainder > 0:
                # Distribute remaining bits
                fractional_parts = best_bits - rounded_bits
                indices = np.argsort(fractional_parts)[-remainder:]
                rounded_bits[indices] += 1
            
            return rounded_bits
            
        except Exception as e:
            print(f"Error in optimization: {str(e)}")
            # Fallback to uniform distribution
            bits_per_band = total_bits // len(variances)
            uniform_bits = np.full(len(variances), bits_per_band, dtype=int)
            remainder = total_bits - np.sum(uniform_bits)
            if remainder > 0:
                uniform_bits[:remainder] += 1
            return uniform_bits

def load_masking_threshold(num_bands):
    """
    Load masking threshold data
    """
    try:
        with open('psychoacoustic_analysis/masking_thresholds.json', 'r') as f:
            masking_data = json.load(f)
        
        available_bands = len(masking_data['band_energies'])
        print(f"Available masking threshold bands: {available_bands}")
        
        # Get available energies
        available_energies = []
        for i in range(available_bands):
            key = f'band_{i}'
            if key in masking_data['band_energies']:
                available_energies.append(masking_data['band_energies'][key])
        
        # Convert to numpy array
        available_energies = np.array(available_energies)
        
        # Interpolate if needed
        if num_bands != available_bands:
            print(f"Interpolating from {available_bands} to {num_bands} bands")
            x_original = np.linspace(0, 1, available_bands)
            x_target = np.linspace(0, 1, num_bands)
            interpolated = np.interp(x_target, x_original, available_energies)
            return interpolated
        
        return available_energies
        
    except Exception as e:
        print(f"Error loading masking thresholds: {str(e)}")
        return np.full(num_bands, -60)

def load_bit_allocation_results(num_bands):
    """
    Load bit allocation results
    """
    try:
        file_path = f'bit_allocation_{num_bands}/allocation_results.json'
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading bit allocation results: {str(e)}")
        return None

def analyze_and_optimize(configurations=[64, 128, 512]):
    main_output_dir = 'lagrange_optimization_analysis'
    os.makedirs(main_output_dir, exist_ok=True)
    
    results = {}
    
    for num_bands in configurations:
        print(f"\nProcessing {num_bands} bands configuration...")
        
        try:
            # Create output directory
            output_dir = os.path.join(main_output_dir, f'analysis_{num_bands}bands')
            os.makedirs(output_dir, exist_ok=True)
            
            # Load data
            bit_alloc = load_bit_allocation_results(num_bands)
            if bit_alloc is None:
                print(f"Skipping {num_bands} bands due to missing data")
                continue
                
            masking_data = load_masking_threshold(num_bands)
            
            # Initialize analyzers
            loss_analyzer = PsychoacousticLossAnalyzer(num_bands)
            optimizer = LagrangeOptimizer(num_bands)
            
            # Process each bit depth
            bit_depths = ['original', '24bit', '16bit']
            analysis_results = {}
            
            for bit_depth in bit_depths:
                print(f"\nAnalyzing {bit_depth}...")
                
                # Get data for current bit depth
                current_variances = np.array(bit_alloc[bit_depth]['variances'])
                current_bits = np.array(bit_alloc[bit_depth]['allocated_bits'])
                total_bits = int(np.sum(current_bits))
                
                # Compute loss
                metrics, error, masked_error = loss_analyzer.compute_spectral_loss(
                    current_variances,
                    current_variances,  # Compare with itself for baseline
                    masking_data
                )
                
                # Optimize bit allocation
                optimized_bits = optimizer.optimize_bit_allocation(
                    variances=current_variances,
                    masking_thresholds=masking_data,
                    total_bits=total_bits,
                    psychoacoustic_loss=masked_error
                )
                
                # Store results for this bit depth
                analysis_results[bit_depth] = {
                    'original': {
                        'variances': current_variances.tolist(),
                        'bits': current_bits.tolist()
                    },
                    'optimized': {
                        'bits': optimized_bits.tolist(),
                        'total_bits': int(np.sum(optimized_bits))
                    },
                    'metrics': metrics
                }
                
                # Create plots for this bit depth
                create_analysis_plots(
                    num_bands=num_bands,
                    bit_depth=bit_depth,
                    original_bits=current_bits,
                    optimized_bits=optimized_bits,
                    variances=current_variances,
                    masking_threshold=masking_data,
                    output_dir=output_dir
                )
                
                print(f"Results for {bit_depth} - {num_bands} bands:")
                print(f"Total bits: {total_bits}")
                print(f"Original allocation - Max: {np.max(current_bits):.2f}, " +
                      f"Min: {np.min(current_bits):.2f}, " +
                      f"Mean: {np.mean(current_bits):.2f}")
                print(f"Optimized allocation - Max: {np.max(optimized_bits):.2f}, " +
                      f"Min: {np.min(optimized_bits):.2f}, " +
                      f"Mean: {np.mean(optimized_bits):.2f}")
            
            # Store results for all bit depths
            results[num_bands] = analysis_results
            
            # Save detailed results
            with open(os.path.join(output_dir, 'optimization_results.json'), 'w') as f:
                json.dump(results[num_bands], f, indent=4)
            
        except Exception as e:
            print(f"Error processing {num_bands} bands: {str(e)}")
            continue
    
    return results, main_output_dir

def create_analysis_plots(num_bands, bit_depth, original_bits, optimized_bits, 
                         variances, masking_threshold, output_dir):
    try:
        plt.figure(figsize=(15, 12))
        
        # Bit Allocation Plot
        plt.subplot(311)
        plt.plot(original_bits, 'b-', label=f'Original {bit_depth}')
        plt.plot(optimized_bits, 'r--', label='Optimized')
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Bits')
        plt.title(f'Bit Allocation for {bit_depth} ({num_bands} bands)')
        plt.legend()
        
        # Variance Plot
        plt.subplot(312)
        plt.semilogy(variances, 'g-', label='Variance')
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Variance (log scale)')
        plt.title(f'Subband Variances - {bit_depth}')
        plt.legend()
        
        # Masking Threshold Plot
        plt.subplot(313)
        plt.plot(masking_threshold, 'm-', label='Masking Threshold')
        plt.grid(True)
        plt.xlabel('Subband Index')
        plt.ylabel('Threshold (dB)')
        plt.title('Masking Thresholds')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'analysis_plots_{bit_depth}.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error creating plots for {bit_depth}: {str(e)}")

if __name__ == "__main__":
    try:
        # Run optimization
        results, output_dir = analyze_and_optimize(configurations=[64, 128, 512])
        
        if results:
            print("\nOptimization completed successfully!")
            
            # Create summary
            summary_data = []
            for num_bands, band_results in results.items():
                for bit_depth, data in band_results.items():
                    original_bits = np.array(data['original']['bits'])
                    optimized_bits = np.array(data['optimized']['bits'])
                    
                    summary_data.append({
                        'Num_Bands': num_bands,
                        'Bit_Depth': bit_depth,
                        'Total_Bits': data['optimized']['total_bits'],
                        'Original_Max': np.max(original_bits),
                        'Optimized_Max': np.max(optimized_bits),
                        'Original_Mean': np.mean(original_bits),
                        'Optimized_Mean': np.mean(optimized_bits),
                        'MSE': data['metrics']['mse'],
                        'Masked_MSE': data['metrics']['masked_mse']
                    })
            
            # Save summary
            df = pd.DataFrame(summary_data)
            df.to_csv(os.path.join(output_dir, 'optimization_summary.csv'), index=False)
            print("\nOptimization Summary:")
            print(df.to_string(index=False))
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

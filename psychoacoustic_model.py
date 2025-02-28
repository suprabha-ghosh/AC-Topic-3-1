import numpy as np
from scipy.fftpack import fft
from scipy.signal import windows
import pandas as pd
import os
import matplotlib.pyplot as plt
from signal_analysis import SignalAnalysis

class PsychoacousticModel:
    def __init__(self, sample_rate):
        """
        Initialize psychoacoustic model
        
        Args:
            sample_rate (int): Audio sampling rate
        """
        self.sample_rate = sample_rate
        self.bark_params = {
            'alpha': 13,
            'beta': 3.5,
            'freq_min': 20,
            'freq_max': 20000
        }
        self.absolute_threshold = {
            'a': 3.64,
            'b': -0.8,
            'c': -6.5,
            'threshold_quiet': -30
        }
        self._init_frequency_tables()

    def _init_frequency_tables(self):
        """Initialize frequency-related lookup tables"""
        self.fft_size = 2048
        self.freq_bins = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)
        self.bark_scale = self._frequency_to_bark(np.abs(self.freq_bins))
        
        # Initialize critical bands
        self.critical_bands = np.arange(0, 25)
        self.critical_band_edges = self._get_critical_band_edges()

    def _frequency_to_bark(self, freq):
        """
        Convert frequency to Bark scale
        
        Args:
            freq (np.ndarray): Frequency values in Hz
            
        Returns:
            np.ndarray: Bark scale values
        """
        return 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500.0) ** 2)

    def _bark_to_frequency(self, bark):
        """
        Convert Bark scale to frequency (approximate inverse)
        
        Args:
            bark (float): Bark value
            
        Returns:
            float: Frequency in Hz
        """
        if bark < 2:
            return 100 * bark
        elif bark < 20.1:
            return 100 * (10 ** (0.05 * bark))
        else:
            return 100 * (10 ** (0.05 * 20.1))

    def _get_critical_band_edges(self):
        """
        Calculate critical band edge frequencies
        
        Returns:
            np.ndarray: Array of critical band edge frequencies
        """
        bark_edges = np.arange(0, 25)
        freq_edges = [self._bark_to_frequency(bark) for bark in bark_edges]
        return np.array(freq_edges)

    def _spreading_function(self, bark_diff):
        """
        Calculate spreading function
        
        Args:
            bark_diff (np.ndarray): Bark scale differences
            
        Returns:
            np.ndarray: Spreading function values
        """
        spread = (15.81 + 7.5 * (bark_diff + 0.474) - 
                 17.5 * np.sqrt(1 + (bark_diff + 0.474) ** 2))
        return np.clip(spread, -100, 0)

    def _absolute_threshold_function(self, freq):
        """
        Compute absolute threshold of hearing
        
        Args:
            freq (np.ndarray): Frequency values in Hz
            
        Returns:
            np.ndarray: Absolute threshold values
        """
        valid_freq = np.maximum(freq, 1e-10)
        threshold = (self.absolute_threshold['a'] * 
                    (valid_freq/1000) ** self.absolute_threshold['b'] + 
                    self.absolute_threshold['c'])
        return np.maximum(threshold, self.absolute_threshold['threshold_quiet'])

    def _find_tonal_maskers(self, power_db):
        """
        Identify tonal maskers in the spectrum
        
        Args:
            power_db (np.ndarray): Power spectrum in dB
            
        Returns:
            dict: Dictionary containing tonal masker information
        """
        # Find local maxima
        local_max = np.zeros_like(power_db, dtype=bool)
        for i in range(1, len(power_db)-1):
            if power_db[i] > power_db[i-1] and power_db[i] > power_db[i+1]:
                local_max[i] = True

        # Apply tonality criteria
        tonal_indices = np.where(local_max)[0]
        tonal_amplitudes = power_db[local_max]
        tonal_bark = self.bark_scale[local_max]

        return {
            'indices': tonal_indices,
            'amplitudes': tonal_amplitudes,
            'bark_values': tonal_bark
        }

    def _find_noise_maskers(self, power_db):
        """
        Identify noise maskers in the spectrum
        
        Args:
            power_db (np.ndarray): Power spectrum in dB
            
        Returns:
            dict: Dictionary containing noise masker information
        """
        bark_bands = np.arange(0, 25)
        noise_amplitudes = np.zeros_like(bark_bands, dtype=float)

        for i, bark_val in enumerate(bark_bands):
            mask = (self.bark_scale >= bark_val) & (self.bark_scale < bark_val + 1)
            if np.any(mask):
                noise_amplitudes[i] = np.mean(power_db[mask])

        return {
            'bark_values': bark_bands,
            'amplitudes': noise_amplitudes
        }

    def _analyze_critical_bands(self, power_db):
        """
        Analyze energy in critical bands
        
        Args:
            power_db (np.ndarray): Power spectrum in dB
            
        Returns:
            np.ndarray: Energy in each critical band
        """
        critical_band_energy = np.zeros(len(self.critical_bands))
        
        for i, bark_val in enumerate(self.critical_bands):
            mask = (self.bark_scale >= bark_val) & (self.bark_scale < bark_val + 1)
            if np.any(mask):
                energy = np.sum(10**(power_db[mask]/10))
                critical_band_energy[i] = 10 * np.log10(energy + 1e-10)
        
        return critical_band_energy

    def compute_masking_threshold(self, frame):
        """
        Compute masking threshold for a single frame
        
        Args:
            frame (np.ndarray): Audio frame
            
        Returns:
            np.ndarray: Masking threshold
        """
        # Normalize frame
        frame = frame / (np.max(np.abs(frame)) + 1e-10)

        # Compute spectrum
        spectrum = fft(frame)
        power_spectrum = np.abs(spectrum) ** 2
        power_db = 10 * np.log10(np.maximum(power_spectrum / np.max(power_spectrum), 1e-10))

        # Find maskers
        tonal_maskers = self._find_tonal_maskers(power_db)
        noise_maskers = self._find_noise_maskers(power_db)

        # Combine masking effects
        masking_threshold = self._combine_masking(tonal_maskers, noise_maskers)

        return masking_threshold

    def _combine_masking(self, tonal_maskers, noise_maskers):
        """
        Combine masking effects from different sources
        
        Args:
            tonal_maskers (dict): Tonal masker information
            noise_maskers (dict): Noise masker information
            
        Returns:
            np.ndarray: Combined masking threshold
        """
        masking_threshold = np.full(self.fft_size, -100.0)

        # Process tonal maskers
        for idx, amp, bark in zip(tonal_maskers['indices'], 
                                tonal_maskers['amplitudes'],
                                tonal_maskers['bark_values']):
            bark_diff = self.bark_scale - bark
            spread = self._spreading_function(bark_diff)
            masking_threshold = np.maximum(masking_threshold, amp + spread)

        # Process noise maskers
        for bark, amp in zip(noise_maskers['bark_values'],
                           noise_maskers['amplitudes']):
            bark_diff = self.bark_scale - bark
            spread = self._spreading_function(bark_diff) + 3
            masking_threshold = np.maximum(masking_threshold, amp + spread)

        # Apply absolute threshold
        absolute_thresh = self._absolute_threshold_function(np.abs(self.freq_bins))
        masking_threshold = np.maximum(masking_threshold, absolute_thresh)
        
        return np.clip(masking_threshold, -100, 0)
    
    def analyze_frames(self, windowed_frames, output_dir="psychoacoustic_analysis"):
        """
        Analyze windowed frames and save results
        
        Args:
            windowed_frames (np.ndarray): Frames from signal analysis
            output_dir (str): Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results storage
        frame_results = []
        masker_results = []
        threshold_results = []
        critical_band_results = []

        # Process each frame
        for frame_idx, frame in enumerate(windowed_frames):
            # Normalize frame
            frame = frame / (np.max(np.abs(frame)) + 1e-10)
            
            # Compute masking threshold
            masking_threshold = self.compute_masking_threshold(frame)
            
            # Get spectrum and maskers
            spectrum = fft(frame)
            power_spectrum = np.abs(spectrum) ** 2
            power_db = 10 * np.log10(np.maximum(power_spectrum / np.max(power_spectrum), 1e-10))
            
            # Find maskers
            tonal_maskers = self._find_tonal_maskers(power_db)
            noise_maskers = self._find_noise_maskers(power_db)

            # Analyze critical bands
            critical_band_energy = self._analyze_critical_bands(power_db)

            # Store frame analysis
            frame_results.append({
                'frame_index': frame_idx,
                'max_threshold': float(np.max(masking_threshold)),
                'min_threshold': float(np.min(masking_threshold)),
                'mean_threshold': float(np.mean(masking_threshold)),
                'num_tonal_maskers': len(tonal_maskers['indices']),
                'num_noise_maskers': len(noise_maskers['bark_values']),
                'total_masking_energy': float(np.sum(10**(masking_threshold/10)))
            })

            # Store masker information
            for i, (idx, amp) in enumerate(zip(tonal_maskers['indices'], 
                                             tonal_maskers['amplitudes'])):
                masker_results.append({
                    'frame_index': frame_idx,
                    'masker_type': 'tonal',
                    'frequency': float(self.freq_bins[idx]),
                    'amplitude': float(amp),
                    'bark_value': float(self.bark_scale[idx])
                })

            # Store critical band analysis
            for band, energy in enumerate(critical_band_energy):
                critical_band_results.append({
                    'frame_index': frame_idx,
                    'critical_band': band,
                    'energy': float(energy),
                    'lower_freq': float(self.critical_band_edges[band]),
                    'upper_freq': float(self.critical_band_edges[band + 1] if band < 24 else self.sample_rate/2)
                })

            # Store threshold values (reduced sampling for efficiency)
            for freq_idx, thresh in enumerate(masking_threshold):
                if freq_idx % 4 == 0:  # Save every 4th point
                    threshold_results.append({
                        'frame_index': frame_idx,
                        'frequency': float(self.freq_bins[freq_idx]),
                        'threshold': float(thresh)
                    })

            # Generate comprehensive plot for select frames
            if frame_idx % 20 == 0:
                self._plot_comprehensive_analysis(frame_idx, power_db, masking_threshold, 
                                               tonal_maskers, noise_maskers, 
                                               critical_band_energy, output_dir)

        # Save results to CSV files
        pd.DataFrame(frame_results).to_csv(
            f"{output_dir}/frame_analysis.csv", index=False)
        pd.DataFrame(masker_results).to_csv(
            f"{output_dir}/masker_analysis.csv", index=False)
        pd.DataFrame(threshold_results).to_csv(
            f"{output_dir}/threshold_analysis.csv", index=False)
        pd.DataFrame(critical_band_results).to_csv(
            f"{output_dir}/critical_band_analysis.csv", index=False)

        # Generate summary plots
        self._plot_summary_analysis(frame_results, critical_band_results, output_dir)

        return output_dir

    def _plot_comprehensive_analysis(self, frame_idx, power_db, masking_threshold, 
                                   tonal_maskers, noise_maskers, critical_band_energy,
                                   output_dir):
        """Generate comprehensive analysis plot"""
        plt.figure(figsize=(15, 10))
        
        # 1. Power Spectrum and Masking (Top Left)
        plt.subplot(2, 2, 1)
        freq = np.abs(self.freq_bins)
        plt.semilogx(freq, power_db, 'b-', label='Power Spectrum', alpha=0.6)
        plt.semilogx(freq, masking_threshold, 'r-', label='Masking Threshold')
        plt.scatter(freq[tonal_maskers['indices']], 
                   tonal_maskers['amplitudes'],
                   color='g', marker='x', label='Tonal Maskers')
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Power Spectrum and Masking')
        plt.legend()
        plt.xlim([20, self.sample_rate/2])
        plt.ylim([-100, 0])

        # 2. Critical Band Energy (Top Right)
        plt.subplot(2, 2, 2)
        plt.bar(self.critical_bands, critical_band_energy, 
                alpha=0.7, label='Band Energy')
        plt.grid(True)
        plt.xlabel('Critical Band')
        plt.ylabel('Energy (dB)')
        plt.title('Critical Band Energy')
        plt.legend()

        # 3. Masking Components (Bottom Left)
        plt.subplot(2, 2, 3)
        plt.plot(freq, masking_threshold, 'r-', label='Combined Threshold')
        absolute_thresh = self._absolute_threshold_function(freq)
        plt.plot(freq, absolute_thresh, 'k--', label='Absolute Threshold')
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Masking Components')
        plt.legend()
        plt.ylim([-100, 0])

        # 4. Bark Scale Analysis (Bottom Right)
        plt.subplot(2, 2, 4)
        plt.plot(self.bark_scale, power_db, 'b-', alpha=0.3, label='Power Spectrum')
        if len(tonal_maskers['indices']) > 0:
            bark_values = self.bark_scale[tonal_maskers['indices']]
            plt.scatter(bark_values, tonal_maskers['amplitudes'], 
                       color='g', marker='x', label='Tonal Maskers')
        plt.grid(True)
        plt.xlabel('Bark Scale')
        plt.ylabel('Magnitude (dB)')
        plt.title('Bark Scale Analysis')
        plt.legend()
        plt.ylim([-100, 0])

        plt.suptitle(f'Frame {frame_idx} Comprehensive Analysis', size=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_{frame_idx}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_summary_analysis(self, frame_results, critical_band_results, output_dir):
        """Generate summary analysis plots"""
        plt.figure(figsize=(15, 10))

        # 1. Threshold Statistics Over Time
        plt.subplot(2, 2, 1)
        df_frames = pd.DataFrame(frame_results)
        plt.plot(df_frames['frame_index'], df_frames['max_threshold'], 
                label='Max Threshold')
        plt.plot(df_frames['frame_index'], df_frames['mean_threshold'], 
                label='Mean Threshold')
        plt.plot(df_frames['frame_index'], df_frames['min_threshold'], 
                label='Min Threshold')
        plt.grid(True)
        plt.xlabel('Frame Index')
        plt.ylabel('Threshold (dB)')
        plt.title('Masking Threshold Statistics')
        plt.legend()

        # 2. Number of Maskers Over Time
        plt.subplot(2, 2, 2)
        plt.plot(df_frames['frame_index'], df_frames['num_tonal_maskers'], 
                label='Tonal Maskers')
        plt.grid(True)
        plt.xlabel('Frame Index')
        plt.ylabel('Count')
        plt.title('Number of Maskers')
        plt.legend()

        # 3. Average Critical Band Energy
        plt.subplot(2, 2, 3)
        df_bands = pd.DataFrame(critical_band_results)
        avg_band_energy = df_bands.groupby('critical_band')['energy'].mean()
        plt.bar(avg_band_energy.index, avg_band_energy.values, alpha=0.7)
        plt.grid(True)
        plt.xlabel('Critical Band')
        plt.ylabel('Average Energy (dB)')
        plt.title('Average Critical Band Energy')

        # 4. Total Masking Energy Over Time
        plt.subplot(2, 2, 4)
        plt.plot(df_frames['frame_index'], df_frames['total_masking_energy'])
        plt.grid(True)
        plt.xlabel('Frame Index')
        plt.ylabel('Energy')
        plt.title('Total Masking Energy')

        plt.suptitle('Summary Analysis', size=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/summary_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    try:
        # Initialize signal analyzer and process audio
        signal_analyzer = SignalAnalysis()
        filename = "data/input_32bit.wav"
        
        # Signal analysis
        windowed_frames, sample_rate, signal_output_dir = signal_analyzer.process_audio(
            filename, "signal_analysis")
        
        if windowed_frames is not None:
            # Psychoacoustic analysis
            psych_model = PsychoacousticModel(sample_rate)
            psych_output_dir = psych_model.analyze_frames(
                windowed_frames, "psychoacoustic_analysis")
            
            print("\nAnalysis completed!")
            print(f"Signal analysis results: {signal_output_dir}")
            print(f"Psychoacoustic analysis results: {psych_output_dir}")
            
            # Display results
            psych_analysis = pd.read_csv(f"{psych_output_dir}/frame_analysis.csv")
            print("\nPsychoacoustic Analysis Summary (first 5 frames):")
            print(psych_analysis.head().to_string())
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import os

class SignalAnalysis:
    def __init__(self, frame_size=2048, hop_size=1024):
        self.frame_size = frame_size
        self.hop_size = hop_size

    def load_audio(self, filename):
        """
        Load audio file and convert to mono if stereo
        
        Args:
            filename (str): Path to audio file
            
        Returns:
            tuple: (signal, sample_rate)
        """
        try:
            signal, sample_rate = sf.read(filename, dtype='float32')
            
            # Convert to mono if stereo
            if signal.ndim > 1:
                signal = np.mean(signal, axis=1)
            
            # Normalize signal
            signal = signal / np.max(np.abs(signal))
            
            return signal, sample_rate
            
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")

    def frame_segmentation(self, signal):
        """
        Segment signal into overlapping frames
        
        Args:
            signal (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: Array of frames
        """
        num_frames = (len(signal) - self.frame_size) // self.hop_size + 1
        frames = np.zeros((num_frames, self.frame_size))
        
        for i in range(num_frames):
            start = i * self.hop_size
            frames[i] = signal[start:start + self.frame_size]
        
        return frames

    def apply_windowing(self, frames):
        """
        Apply Hann window to frames
        
        Args:
            frames (np.ndarray): Input frames
            
        Returns:
            np.ndarray: Windowed frames
        """
        window = np.hanning(self.frame_size)
        return frames * window

    def analyze_frame(self, frame, frame_index, sample_rate):
        """
        Analyze a single frame
        
        Args:
            frame (np.ndarray): Audio frame
            frame_index (int): Frame index
            sample_rate (int): Sampling rate
            
        Returns:
            dict: Frame analysis results
        """
        # Time domain analysis
        rms = np.sqrt(np.mean(frame**2))
        peak = np.max(np.abs(frame))
        energy = np.sum(frame**2)
        zero_crossings = np.sum(np.diff(np.signbit(frame)))
        
        # Frequency domain analysis
        spectrum = np.fft.fft(frame)
        frequencies = np.fft.fftfreq(len(frame), 1/sample_rate)
        magnitudes = np.abs(spectrum)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(magnitudes[:len(magnitudes)//2])
        dominant_freq = abs(frequencies[dominant_freq_idx])
        
        return {
            'frame_index': frame_index,
            'time_start': frame_index * self.hop_size / sample_rate,
            'rms': rms,
            'peak': peak,
            'energy': energy,
            'zero_crossings': zero_crossings,
            'dominant_frequency': dominant_freq,
            'spectral_centroid': np.sum(frequencies * magnitudes) / np.sum(magnitudes),
            'spectral_spread': np.sqrt(np.sum(((frequencies - np.mean(frequencies))**2) * magnitudes) / np.sum(magnitudes))
        }

    def analyze_signal(self, signal, sample_rate):
        """
        Analyze entire signal
        
        Args:
            signal (np.ndarray): Input signal
            sample_rate (int): Sampling rate
            
        Returns:
            dict: Signal analysis results
        """
        return {
            'duration': len(signal) / sample_rate,
            'num_samples': len(signal),
            'sample_rate': sample_rate,
            'max_amplitude': np.max(np.abs(signal)),
            'min_amplitude': np.min(signal),
            'mean_amplitude': np.mean(signal),
            'rms_amplitude': np.sqrt(np.mean(signal**2)),
            'total_energy': np.sum(signal**2),
            'zero_crossings_rate': np.sum(np.diff(np.signbit(signal))) / len(signal)
        }

    def save_analysis_plots(self, signal, frames, windowed_frames, sample_rate, output_dir):
        """
        Generate and save analysis plots
        
        Args:
            signal (np.ndarray): Input signal
            frames (np.ndarray): Frame array
            windowed_frames (np.ndarray): Windowed frames
            sample_rate (int): Sampling rate
            output_dir (str): Output directory
        """
        # 1. Waveform
        plt.figure(figsize=(12, 6))
        time = np.arange(len(signal)) / sample_rate
        plt.plot(time, signal)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.savefig(f"{output_dir}/waveform.png")
        plt.close()

        # 2. Spectrogram
        plt.figure(figsize=(12, 6))
        plt.specgram(signal, Fs=sample_rate, NFFT=self.frame_size)
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Intensity (dB)')
        plt.savefig(f"{output_dir}/spectrogram.png")
        plt.close()

        # 3. Frame Energy
        plt.figure(figsize=(12, 4))
        frame_energies = np.array([np.sum(frame**2) for frame in frames])
        plt.plot(frame_energies)
        plt.title('Frame Energy')
        plt.xlabel('Frame Index')
        plt.ylabel('Energy')
        plt.grid(True)
        plt.savefig(f"{output_dir}/frame_energies.png")
        plt.close()

        # 4. Average Spectrum
        plt.figure(figsize=(12, 6))
        avg_spectrum = np.mean(np.abs(np.fft.fft(windowed_frames, axis=1)), axis=0)
        freqs = np.fft.fftfreq(self.frame_size, 1/sample_rate)
        plt.plot(freqs[:len(freqs)//2], avg_spectrum[:len(freqs)//2])
        plt.title('Average Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.savefig(f"{output_dir}/average_spectrum.png")
        plt.close()

    def process_audio(self, filename, output_dir="signal_analysis"):
        """
        Main processing function
        
        Args:
            filename (str): Input audio file path
            output_dir (str): Output directory for results
            
        Returns:
            tuple: (windowed_frames, sample_rate, output_directory)
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Load audio
            signal, sample_rate = self.load_audio(filename)

            # Analyze signal
            signal_analysis = self.analyze_signal(signal, sample_rate)
            pd.DataFrame([signal_analysis]).to_csv(
                f"{output_dir}/signal_analysis.csv", index=False)

            # Frame processing
            frames = self.frame_segmentation(signal)
            windowed_frames = self.apply_windowing(frames)

            # Analyze frames
            frame_analysis = []
            for i in range(len(frames)):
                analysis = self.analyze_frame(frames[i], i, sample_rate)
                frame_analysis.append(analysis)

            # Save frame analysis
            pd.DataFrame(frame_analysis).to_csv(
                f"{output_dir}/frame_analysis.csv", index=False)

            # Generate plots
            self.save_analysis_plots(signal, frames, windowed_frames, 
                                   sample_rate, output_dir)

            return windowed_frames, sample_rate, output_dir

        except Exception as e:
            print(f"Error in signal analysis: {str(e)}")
            return None, None, None

def main():
    try:
        # Initialize analyzer
        analyzer = SignalAnalysis()
        
        # Process audio
        filename = "data/input_32bit.wav"
        output_dir = "signal_analysis"
        windowed_frames, sample_rate, output_dir = analyzer.process_audio(filename, output_dir)
        
        if windowed_frames is not None:
            print(f"\nSignal analysis completed successfully!")
            print(f"Results saved in: {output_dir}")
            
            # Display some results
            signal_analysis = pd.read_csv(f"{output_dir}/signal_analysis.csv")
            frame_analysis = pd.read_csv(f"{output_dir}/frame_analysis.csv")
            
            print("\nSignal Analysis Summary:")
            print(signal_analysis.to_string())
            print("\nFrame Analysis Summary (first 5 frames):")
            print(frame_analysis.head().to_string())
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

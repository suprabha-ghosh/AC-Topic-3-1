import numpy as np
import soundfile as sf
import os
from psychoacoustic_model import PsychoacousticModel
from subband_analysis import SubbandAnalysis
from bit_allocation import BitAllocation
from quantizer import Quantizer
from loss_function import PsychoacousticLoss

def process_audio(input_file, num_subbands, target_bitrate, frame_size=2048):
    audio, sample_rate = sf.read(input_file, dtype='float32')
    
    psych_model = PsychoacousticModel(sample_rate)
    subband_analyzer = SubbandAnalysis(num_subbands, frame_size)
    bit_allocator = BitAllocation(target_bitrate)
    quantizer = Quantizer(target_bitrate)  # Pass target_bitrate here
    loss_calculator = PsychoacousticLoss(sample_rate)  # Pass sample_rate here
    
    num_frames = len(audio) // frame_size
    quantized_audio = np.zeros_like(audio[:num_frames * frame_size])
    total_loss = 0
    
    for i in range(num_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        masking_threshold = psych_model.compute_masking_threshold(frame)
        subband_signals = subband_analyzer.analyze_subbands(frame)
        
        energies = bit_allocator.compute_subband_energies(subband_signals)
        bit_allocation = bit_allocator.optimize_allocation(
            energies, masking_threshold, frame_size
        )
        
        quantized_subbands = quantizer.quantize_subbands(
            subband_signals, bit_allocation, masking_threshold
        )
        
        reconstructed_frame = subband_analyzer.synthesize_subbands(quantized_subbands)
        loss_metrics = loss_calculator.compute_loss(
            frame, reconstructed_frame, masking_threshold
        )
        total_loss += loss_metrics['weighted_error']  # Use weighted_error from metrics
        
        quantized_audio[i * frame_size:(i + 1) * frame_size] = reconstructed_frame
    
    average_loss = total_loss / num_frames
    print(f"Target {target_bitrate}-bit - Average psychoacoustic loss: {average_loss:.2f} dB")
    
    return quantized_audio, sample_rate

def main():
    INPUT_FILE = "data/input_32bit.wav"
    OUTPUT_DIR = "data/output/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    subband_configs = [64, 128, 512, 1024, 2048]
    bit_depths = {
        '24bit': 24,
        '16bit': 16
    }
    
    for num_subbands in subband_configs:
        print(f"\nProcessing with {num_subbands} subbands...")
        
        for bit_depth_name, target_bitrate in bit_depths.items():
            print(f"\nConverting to {bit_depth_name}...")
            
            try:
                quantized_audio, sample_rate = process_audio(
                    INPUT_FILE,
                    num_subbands=num_subbands,
                    target_bitrate=target_bitrate
                )
                
                output_file = f"{OUTPUT_DIR}quantized_{num_subbands}subbands_{bit_depth_name}.wav"
                
                if bit_depth_name == '24bit':
                    sf.write(output_file, quantized_audio, sample_rate, subtype='PCM_24')
                else:  # 16bit
                    sf.write(output_file, quantized_audio, sample_rate, subtype='PCM_16')
                    
                print(f"Saved {bit_depth_name} output to {output_file}")
                
            except Exception as e:
                print(f"Error processing {bit_depth_name} with {num_subbands} subbands: {str(e)}")
                continue

if __name__ == "__main__":
    main()

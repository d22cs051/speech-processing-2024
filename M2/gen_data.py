# question 1: Create an audio dataset of 100 examples for source separation such that source1 is a sinusoid
# with a random frequency smaller than fundamental frequency fs =8000, and source2 is a sinusoid with a
# frequency larger than fs. The length of time steps (T) is 10000.


# ref: https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python
# import time
# import numpy as np
# import pyaudio
# import wave


# p = pyaudio.PyAudio()

# volume = 0.5  # range [0.0, 1.0]
# fs = 8000  # sampling rate, Hz, must be integer
# duration = 1.25  # in seconds, may be float => T=10000
# f = 440.0  # sine frequency, Hz, may be float

# # generate samples, note conversion to float32 array
# # print(f"Generating samples, len: {len(np.arange(fs * duration))}")
# samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

# # per @yahweh comment explicitly convert to bytes sequence
# output_bytes = (volume * samples).tobytes()

# # for paFloat32 sample values must be in range [-1.0, 1.0]
# stream = p.open(format=pyaudio.paFloat32,
#                 channels=1,
#                 rate=fs,
#                 output=True)

# # play. May repeat with different volume values (if done interactively)
# start_time = time.time()
# stream.write(output_bytes)
# # Save audio signal as WAV file
# i=0
# wave_file = wave.open(f"audio_{i}.wav", 'wb')
# wave_file.setnchannels(1)
# wave_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
# wave_file.setframerate(fs)
# wave_file.writeframes(output_bytes)
# wave_file.close()
# print("Played sound for {:.2f} seconds".format(time.time() - start_time))

# stream.stop_stream()
# stream.close()

# p.terminate()

import numpy as np
import pyaudio
import wave
from pathlib import Path

def generate_and_save_audio_example(fs=8000, duration=1.25, f1=2000.0, f2=10000.0, num_samples=0):
    p = pyaudio.PyAudio()

    volume = 0.2  # range [0.0, 1.0]

    print(f"Generating datapoints, len: {len(np.arange(fs * duration))}")

    # generate source1 samples
    source1 = (np.sin(2 * np.pi * np.arange(fs * duration) * f1 / fs)).astype(np.float32)

    # generate source2 samples
    source2 = (np.sin(2 * np.pi * np.arange(fs * duration) * f2 / fs)).astype(np.float32)

    # generate mixture
    mixture = source1 + source2

    # per @yahweh comment explicitly convert to bytes sequence
    source1_bytes = (volume * source1).tobytes()
    source2_bytes = (volume * source2).tobytes()
    mixture_bytes = (volume * mixture).tobytes()

    # Save source1 as WAV file
    with wave.open(str(save_dir_s1 / f'sample_{num_samples}_source1_{fs}_{f1}.wav'), 'wb') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(fs)
        wave_file.writeframes(source1_bytes)

    # Save source2 as WAV file
    with wave.open(str(save_dir_s2 / f'sample_{num_samples}_source2_{fs}_{f2}.wav'), 'wb') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(fs)
        wave_file.writeframes(source2_bytes)

    # Save mixture as WAV file
    with wave.open(str(save_dir_mix / f'sample_{num_samples}_mixture_{fs}_{f1}_{f2}.wav'), 'wb') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(fs)
        wave_file.writeframes(mixture_bytes)

    p.terminate()
    
    # adding info in a csv file
    with open(data_root / "info.csv", "a") as f:
        if num_samples == 0:
            f.write("audio_src1, audio_src2, mixture, fs, f1, f2\n")
        # audio src1, audio src2, mixture, fs, f1, f2
        f.write(f"sample_{num_samples}_source1_{fs}_{f1}.wav,sample_{num_samples}_source2_{fs}_{f2}.wav,sample_{num_samples}_mixture_{fs}_{f1}_{f2}.wav,{fs},{f1},{f2}\n")
        

if __name__ == "__main__":
    # Create directories if they don't exist
    data_root = Path("data")
    save_dir_s1 = data_root/Path("source1")
    save_dir_s2 = data_root/Path("source2")
    save_dir_mix = data_root/Path("mixture")
    save_dir_s1.mkdir(parents=True, exist_ok=True)
    save_dir_s2.mkdir(parents=True, exist_ok=True)
    save_dir_mix.mkdir(parents=True, exist_ok=True)

    for i in range(100):
        generate_and_save_audio_example(num_samples=i)

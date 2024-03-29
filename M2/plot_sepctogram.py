import numpy as np
import librosa
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from pathlib import Path

def plot_spectrogram(wave_path:str):
    # loading the audio file
    y, sr = librosa.load(wave_path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting the spectrogram
    spec, freqs, t, im = axs[0].specgram(y, NFFT=2048, Fs=sr, cmap='inferno', scale='dB')
    axs[0].set_title('Spectrogram')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    axs[0].set_ylim(0, 8000)  # Set the y-axis limits to match the frequency range

    # Plotting the mel spectrogram
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time', ax=axs[1])
    axs[1].set_title('Mel Spectrogram')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Mel Frequency')
    axs[1].set_ylim(0, 8000)  # Set the y-axis limits to match the mel frequency range

    # Add frequency meter on the left side of the plots
    axs[0].yaxis.set_label_coords(-0.1, 0.5)
    axs[1].yaxis.set_label_coords(-0.1, 0.5)


    plt.tight_layout()
    # print(wave_path.as_posix().split("/"))
    (save_dir / "/".join(wave_path.as_posix().split("/")[-3:-1])).mkdir(exist_ok=True, parents=True)
    plt.savefig(f'data/plots/{"/".join(wave_path.as_posix().split("/")[-3:]).replace(".wav","")}.png')
    plt.show()


save_dir = Path("data/plots/")

save_dir.mkdir(exist_ok=True, parents=True)
for wav_file in tqdm(Path("data/output/source1").glob("*.wav"),total=100):
    plot_spectrogram(wav_file)
for wav_file in tqdm(Path("data/output/source2").glob("*.wav"),total=100):
    plot_spectrogram(wav_file)
# for wav_file in tqdm(Path("data/mixture").glob("*.wav"),total=100):
#     plot_spectrogram(wav_file)
for wav_file in tqdm(Path("data/output/input1").glob("*.wav"),total=100):
    plot_spectrogram(wav_file)
for wav_file in tqdm(Path("data/output/input2").glob("*.wav"),total=100):
    plot_spectrogram(wav_file)
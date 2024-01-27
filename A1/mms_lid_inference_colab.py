# -*- coding: utf-8 -*-
"""Copy of MMS_LID_Inference_Colab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vvd-M_trfoA9XREAuD4471arJuVM3cCf

# Running MMS-LID inference in Colab

## Step 1: Clone fairseq-py and install latest version
"""

# Commented out IPython magic to ensure Python compatibility.
import os

!git clone https://github.com/pytorch/fairseq

# Change current working directory
!pwd
# %cd "/content/fairseq"
!pip install --editable ./
!pip install tensorboardX

"""## 2. Download MMS-LID model


"""

available_models = ["l126", "l256", "l512", "l1024", "l2048", "l4017"]

# We will use L126 model which can recognize 126 languages
model_name = available_models[0] # l126
print(f"Using model - {model_name}")
print(f"Visit https://dl.fbaipublicfiles.com/mms/lid/mms1b_{model_name}_langs.html to check all the languages supported by this model.")

! mkdir -p /content/models_lid
!wget -P /content/models_lid/{model_name} 'https://dl.fbaipublicfiles.com/mms/lid/mms1b_{model_name}.pt'
!wget -P /content/models_lid/{model_name} 'https://dl.fbaipublicfiles.com/mms/lid/dict/l126/dict.lang.txt'

"""## 3. Prepare manifest files
Create a folder on path '/content/audio_samples/' and upload your .wav audio files that you need to recognize e.g. '/content/audio_samples/abc.wav' , '/content/audio_samples/def.wav' etc...

Note: You need to make sure that the audio data you are using has a sample rate of 16kHz You can easily do this with FFMPEG like the example below that converts .mp3 file to .flac and fixing the audio sample rate

Here, we use three examples - one audio file from English, Hindi, Chinese each.
"""

! mkdir -p /content/audio_samples/
# NOT USING THE BELOW DOING IT ON LOCAL
# for key in ["en_us", "hi_in", "cmn_hans_cn"]:
#   !wget -O /content/audio_samples/tmp.mp3 https://datasets-server.huggingface.co/assets/google/fleurs/--/{key}/train/0/audio/audio.mp3
#   !ffmpeg -hide_banner -loglevel error -y -i   /content/audio_samples/tmp.mp3 -ar 16000 /content/audio_samples/{key}.wav

# ! mkdir -p /content/audio_samples/

! mkdir -p /content/manifest/
import os
with open("/content/manifest/dev.tsv", "w") as ftsv, open("/content/manifest/dev.lang", "w") as flang:
  ftsv.write("/\n")

  for fl in os.listdir("/content/audio_samples/"):
    if not fl.endswith(".wav"):
      continue
    audio_path = f"/content/audio_samples/{fl}"
    # duration should be number of samples in audio. For inference, using a random value should be fine.
    duration = 1234
    ftsv.write(f"{audio_path}\t{duration}\n")
    flang.write("eng\n") # This is the "true" language for the audio. For inference, using a random value should be fine.

"""# 4: Run Inference and transcribe your audio(s)

"""

import os

os.environ["PYTHONPATH"] = "/content/fairseq"
os.environ["PREFIX"] = "INFER"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["USER"] = "mms_lid_user"

!python3 examples/mms/lid/infer.py /content/models_lid/{model_name} --path /content/models_lid/{model_name}/mms1b_l126.pt \
  --task audio_classification  --infer-manifest /content/manifest/dev.tsv --output-path /content/manifest/

print("----- INPUT FILES -----")
! tail -n +2 /content/manifest/dev.tsv

print("\n----- TOP-K PREDICTONS WITH SCORE -----")
! cat /content/manifest//predictions.txt


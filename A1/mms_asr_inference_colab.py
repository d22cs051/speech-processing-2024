# -*- coding: utf-8 -*-
"""Copy of MMS_ASR_Inference_Colab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EVjRrONZc_2-GY-PMNy693JWF3WKW8zE

# Running MMS-ASR inference in Colab

In this notebook, we will give an example on how to run simple ASR inference using MMS ASR model.

Credit to epk2112 [(github)](https://github.com/epk2112/fairseq_meta_mms_Google_Colab_implementation)

## Step 1: Clone fairseq-py and install latest version
"""

# Commented out IPython magic to ensure Python compatibility.
!mkdir "temp_dir"
!git clone https://github.com/pytorch/fairseq

# Change current working directory
!pwd
# %cd "/content/fairseq"
!pip install --editable ./
!pip install tensorboardX

"""## 2. Download MMS model
Un-comment to download your preferred model.
In this example, we use MMS-FL102 for demo purposes.
For better model quality and language coverage, user can use MMS-1B-ALL model instead (but it would require more RAM, so please use Colab-Pro instead of Colab-Free).

"""

# MMS-1B:FL102 model - 102 Languages - FLEURS Dataset
!wget -P ./models_new 'https://dl.fbaipublicfiles.com/mms/asr/mms1b_fl102.pt'

# # MMS-1B:L1107 - 1107 Languages - MMS-lab Dataset
# !wget -P ./models_new 'https://dl.fbaipublicfiles.com/mms/asr/mms1b_l1107.pt'

# # MMS-1B-all - 1162 Languages - MMS-lab + FLEURS + CV + VP + MLS
# !wget -P ./models_new 'https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt'

"""## 3. Prepare audio file
Create a folder on path '/content/audio_samples/' and upload your .wav audio files that you need to transcribe e.g. '/content/audio_samples/audio.wav'

Note: You need to make sure that the audio data you are using has a sample rate of 16kHz You can easily do this with FFMPEG like the example below that converts .mp3 file to .wav and fixing the audio sample rate

Here, we use a FLEURS english MP3 audio for the example.
"""

!mkdir -p "/content/audio_samples"

# Uploading own samples to "/content/audio_samples"
# !wget -P ./audio_samples/ 'https://datasets-server.huggingface.co/assets/google/fleurs/--/en_us/train/0/audio/audio.mp3'
# !ffmpeg -y -i ./audio_samples/audio.mp3 -ar 16000 ./audio_samples/audio.wav

"""# 4: Run Inference and transcribe your audio(s)

In the below example, we will transcribe a sentence in English.

To transcribe other languages:
1. Go to [MMS README ASR section](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr)
2. Open Supported languages link
3. Find your target languages based on Language Name column
4. Copy the corresponding Iso Code
5. Replace `--lang "eng"` with new Iso Code

To improve the transcription quality, user can use language-model (LM) decoding by following this instruction [ASR LM decoding](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr)
"""

import os

os.environ["TMPDIR"] = '/content/temp_dir'
os.environ["PYTHONPATH"] = "."
os.environ["PREFIX"] = "INFER"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["USER"] = "micro"

# !python examples/mms/asr/infer/mms_infer.py --model "/content/fairseq/models_new/mms1b_fl102.pt" --lang "hin" --audio "/content/audio_samples/hin1.wav"
!python examples/mms/asr/infer/mms_infer.py --model "/content/fairseq/models_new/mms1b_fl102.pt" --lang "hin" --audio "/content/audio_samples/gen_hin1.wav"
!python examples/mms/asr/infer/mms_infer.py --model "/content/fairseq/models_new/mms1b_fl102.pt" --lang "hin" --audio "/content/audio_samples/gen_hin2.wav"
!python examples/mms/asr/infer/mms_infer.py --model "/content/fairseq/models_new/mms1b_fl102.pt" --lang "eng" --audio "/content/audio_samples/gen_eng1.wav"
!python examples/mms/asr/infer/mms_infer.py --model "/content/fairseq/models_new/mms1b_fl102.pt" --lang "eng" --audio "/content/audio_samples/gen_eng2.wav"

"""# Computing metrics"""

!pip3 install torchmetrics -q

"""## Character Error Rate (CER)"""

# English
from torchmetrics.text import CharErrorRate

cer = CharErrorRate()

preds = ["te sun shines britly in de steer esu sky kasting ebes golden rave upon te ert illuminated te world with ex rediont vove and splonder", "ol ok tre stud majisticly in te hrt of de forest its gorned branches weching towords de sky prowiding shelder to mercles of crecrs dat coled te wod der hom"]
target = ["The sun shines brightly in the clear azure sky, casting its golden rays upon the earth, illuminating the world with its radiant warmth and splendor.", "The old oak tree stood majestically in the heart of the forest, its gnarled branches reaching towards the sky, providing shelter to a myriad of creatures that called the woods their home."]

print(f"English ASR vs GT, CER: {cer(preds, target)}")

preds_gen = ["the sun shines brightly in the clear as your sky casting its golden rase upon the earth illuminating the world with its radient warmth and splender", "the oldupe tree stood magestically in the heart of the forest its maral branches reaching towards the sky providing shelter to a meriad of creatures that call the woods their home"]
print(f"Gen. English ASR vs GT, CER: {cer(preds_gen, target)}")

# Hindi
from torchmetrics.text import CharErrorRate

cer = CharErrorRate()

preds = ["औफ़ नूले आसमान में सूरज चमकता है अपनी सोने जैसी किरणों को धरती पर गिरहते हुए अपनी तेज़ गर्मी और शांत से दुनिया को रोश्नी में ले जाता है", " र बलूत कापिल जंगल के दिल् में शानदार रूपों में खाना था उसके ऊलझे हुए डालों में आकाश की ओर भढते हुए जंगल के वह अनगिनत प्राणियों को आश्रप्रदान किया जो जंगल को अपना घर कहते थे"]
target = ["साफ नीले आसमान में सूरज चमकता है, अपनी सोने जैसी किरणों को धरती पर गिराते हुए, अपनी तेज गर्मी और शान से दुनिया को रोशनी में ले जाता है।", "पुराना बलूत का पेड़ जंगल के दिल में शानदार रूप में खड़ा था, उसके उलझे हुए डालों ने आकाश की ओर बढ़ते हुए, जंगल के वो अनगिनत प्राणियों को आश्रय प्रदान किया जो जंगल को अपना घर कहते थे।"]

print(f"Hindi ASR vs GT, CER: {cer(preds, target)}")

preds_gen = ["साफ़ नीय आसमान में सूरज चमकता है अपनी सोने जैसी किरणों को धरती पर गिराते हुए अपनी तेज गर्मी और शान से दुनिया को रोश्नी में ले जाता है", "पुराना बलूत का पेड जंगल के दिल में शानदारऔर रूप में खरा था उसके उल्जे हुए डालों ने आकाश की ओर बढ़ते हुए जंडल के वो अनगिनित प्राणियों को आश्य प्रदान किया जो जंकव को अपना घर कहते थे"]
print(f"Gen. Hindi ASR vs GT, CER: {cer(preds_gen, target)}")

"""## Word Error Rate (WER)"""

# English
from torchmetrics.text import WordErrorRate

wer = WordErrorRate()

preds = ["te sun shines britly in de steer esu sky kasting ebes golden rave upon te ert illuminated te world with ex rediont vove and splonder", "ol ok tre stud majisticly in te hrt of de forest its gorned branches weching towords de sky prowiding shelder to mercles of crecrs dat coled te wod der hom"]
target = ["The sun shines brightly in the clear azure sky, casting its golden rays upon the earth, illuminating the world with its radiant warmth and splendor.", "The old oak tree stood majestically in the heart of the forest, its gnarled branches reaching towards the sky, providing shelter to a myriad of creatures that called the woods their home."]

print(f"English ASR vs GT, WER: {wer(preds, target)}")

preds_gen = ["the sun shines brightly in the clear as your sky casting its golden rase upon the earth illuminating the world with its radient warmth and splender", "the oldupe tree stood magestically in the heart of the forest its maral branches reaching towards the sky providing shelter to a meriad of creatures that call the woods their home"]
print(f"Gen. English ASR vs GT, WER: {wer(preds_gen, target)}")

# Hindi
from torchmetrics.text import WordErrorRate

wer = WordErrorRate()

preds = ["औफ़ नूले आसमान में सूरज चमकता है अपनी सोने जैसी किरणों को धरती पर गिरहते हुए अपनी तेज़ गर्मी और शांत से दुनिया को रोश्नी में ले जाता है", " र बलूत कापिल जंगल के दिल् में शानदार रूपों में खाना था उसके ऊलझे हुए डालों में आकाश की ओर भढते हुए जंगल के वह अनगिनत प्राणियों को आश्रप्रदान किया जो जंगल को अपना घर कहते थे"]
target = ["साफ नीले आसमान में सूरज चमकता है, अपनी सोने जैसी किरणों को धरती पर गिराते हुए, अपनी तेज गर्मी और शान से दुनिया को रोशनी में ले जाता है।", "पुराना बलूत का पेड़ जंगल के दिल में शानदार रूप में खड़ा था, उसके उलझे हुए डालों ने आकाश की ओर बढ़ते हुए, जंगल के वो अनगिनत प्राणियों को आश्रय प्रदान किया जो जंगल को अपना घर कहते थे।"]

print(f"Hindi ASR vs GT, WER: {wer(preds, target)}")

preds_gen = ["साफ़ नीय आसमान में सूरज चमकता है अपनी सोने जैसी किरणों को धरती पर गिराते हुए अपनी तेज गर्मी और शान से दुनिया को रोश्नी में ले जाता है", "पुराना बलूत का पेड जंगल के दिल में शानदारऔर रूप में खरा था उसके उल्जे हुए डालों ने आकाश की ओर बढ़ते हुए जंडल के वो अनगिनित प्राणियों को आश्य प्रदान किया जो जंकव को अपना घर कहते थे"]
print(f"Gen. Hindi ASR vs GT, WER: {wer(preds_gen, target)}")

"""# 5: Beam search decoding using a Language Model and transcribe audio file(s)

Since MMS is a CTC model, we can further improve the accuracy by running beam search decoding using a language model.

While we have not open sourced the language models used in MMS (yet!), we have provided the details of the data and commands to used to train the LMs in the Appendix section of our paper.


For this tutorial, we will use a alternate English language model based on Common Crawl data which has been made publicly available through the efforts of [Likhomanenko, Tatiana, et al. "Rethinking evaluation in asr: Are our models robust enough?."](https://arxiv.org/abs/2010.11745). The language model can be accessed from the GitHub repository [here](https://github.com/flashlight/wav2letter/tree/main/recipes/rasr).
"""

! mkdir -p /content/lmdecode

!wget -P /content/lmdecode  https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lm_common_crawl_small_4gram_prun0-6-15_200kvocab.bin # smaller LM
!wget -P /content/lmdecode  https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lexicon.txt

"""
Install decoder bindings from [flashlight](https://github.com/flashlight/flashlight)
"""

# Taken from https://github.com/flashlight/flashlight/blob/main/scripts/colab/colab_install_deps.sh
# Install dependencies from apt
! sudo apt-get install -y libfftw3-dev libsndfile1-dev libgoogle-glog-dev libopenmpi-dev libboost-all-dev
# Install Kenlm
! cd /tmp && git clone https://github.com/kpu/kenlm && cd kenlm && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make install -j$(nproc)

# Install Intel MKL 2020
! cd /tmp && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
! sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends intel-mkl-64bit-2020.0-088
# Remove existing MKL libs to avoid double linkeage
! rm -rf /usr/local/lib/libmkl*

# Commented out IPython magic to ensure Python compatibility.
! rm -rf flashlight
! git clone --recursive https://github.com/flashlight/flashlight.git
# %cd flashlight
! git checkout 035ead6efefb82b47c8c2e643603e87d38850076
# %cd bindings/python
! python3 setup.py install

# %cd /content/fairseq

"""Next, we download an audio file from [People's speech](https://huggingface.co/datasets/MLCommons/peoples_speech) data. We will the audio sample from their 'dirty' subset which will be more challenging for the ASR model."""

!wget -O ./audio_samples/tmp.wav 'https://datasets-server.huggingface.co/assets/MLCommons/peoples_speech/--/dirty/train/0/audio/audio.wav'
!ffmpeg -y -i ./audio_samples/tmp.wav -ar 16000 ./audio_samples/audio_noisy.wav

"""Let's listen to the audio file

"""

import IPython
IPython.display.display(IPython.display.Audio("./audio_samples/audio_noisy.wav"))
print("Trancript: limiting emotions that we experience mainly in our childhood which stop us from living our life just open freedom i mean trust and")

"""Run inference with both greedy decoding and LM decoding"""

import os

os.environ["TMPDIR"] = '/content/temp_dir'
os.environ["PYTHONPATH"] = "."
os.environ["PREFIX"] = "INFER"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["USER"] = "micro"

print("======= WITHOUT LM DECODING=======")

!python examples/mms/asr/infer/mms_infer.py --model "/content/fairseq/models_new/mms1b_fl102.pt" --lang "eng" --audio "/content/fairseq/audio_samples/audio.wav" "/content/fairseq/audio_samples/audio_noisy.wav"

print("\n\n\n======= WITH LM DECODING=======")

# Note that the lmweight, wordscore needs to tuned for each LM
# Using the same values may not be optimal
decoding_cmds = """
decoding.type=kenlm
decoding.beam=500
decoding.beamsizetoken=50
decoding.lmweight=2.69
decoding.wordscore=2.8
decoding.lmpath=/content/lmdecode/lm_common_crawl_small_4gram_prun0-6-15_200kvocab.bin
decoding.lexicon=/content/lmdecode/lexicon.txt
""".replace("\n", " ")
!python examples/mms/asr/infer/mms_infer.py --model "/content/fairseq/models_new/mms1b_fl102.pt" --lang "eng" --audio "/content/fairseq/audio_samples/audio.wav" "/content/fairseq/audio_samples/audio_noisy.wav" \
    --extra-infer-args '{decoding_cmds}'


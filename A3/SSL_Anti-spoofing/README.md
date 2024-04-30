Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation
===============
This if fork for the orginal repo for assignment purpose only.
## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
$ conda create -n SSL_Spoofing python=3.8.10
$ conda activate SSL_Spoofing
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
(This fairseq folder can also be downloaded from https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)
$ pip install --editable ./
$ pip install -r requirements.txt
```


## Pre-trained wav2vec 2.0 XLSR (300M)
Download the XLSR models from [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

We also provide a pre-trained models. To use it you can run: 

Pre-trained SSL antispoofing models are available for LA and DF [here](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB?usp=sharing)


## Running all the question part for custom dataset and FOR dataset
```
python3 main_assign.py
```

## Using online Demo
head over to [https://huggingface.co/spaces/d22cs051/Audio-Deepfake-Detection](https://huggingface.co/spaces/d22cs051/Audio-Deepfake-Detection) for online demo.

## Running Demo

Note: please download all the model file from [here](https://huggingface.co/spaces/d22cs051/Audio-Deepfake-Detection/tree/main/models) and place in model folders.
### using vitual env
```
cd Audio-Deepfake-Detection
pip3 install gradio
python3 app.py
```

### using docker
```
cd Audio-Deepfake-Detection
docker build . -t df:test; docker run -p 7860:7860 -it df:test
```



## Cite the orignial paper if using this code
@inproceedings{tak2022automatic,
  title={Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation},
  author={Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle={The Speaker and Language Recognition Workshop},
  year={2022}
}
```


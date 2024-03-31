# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get install sox -y

WORKDIR /code

COPY ./sp.yml /code/sp.yml

RUN conda env create -f sp.yml
RUN conda activate sp
# RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app
ENV TZ=India/Kolkata

CMD ["python3", "app.py"]
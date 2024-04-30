FROM python:3.8.1-slim

RUN apt-get update
RUN apt-get install ffmpeg build-essential -y

# create a new user
# RUN useradd -m -u 1000 user

# Switch to the "user" user
# USER user

# Set home to the user's home directory
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
# WORKDIR $HOME/code

WORKDIR /code
# COPY . $HOME/code
COPY . .

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# WORKDIR fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 
RUN pip install -e fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1/.
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install gradio --no-cache-dir
RUN pip install protobuf==3.20.* --no-cache-dir

# # Switch to the "user" user
# USER user

# # Set home to the user's home directory
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
# WORKDIR $HOME/code

# COPY --chown=user . $HOME/code
# RUN ls -la $HOME/code
# RUN chown -r 777:777 /code
RUN mkdir flagged
RUN chown -R 777:777 flagged
CMD ["python3", "app.py"]
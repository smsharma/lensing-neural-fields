FROM nvidia/cuda:11.3.1-runtime-ubuntu18.04
ENV LANG=C.UTF-8
ENV PIP_NO_CACHE_DIR=off

RUN apt-get -y update && apt-get install -y git vim tree curl unzip xvfb patchelf ffmpeg cmake swig
RUN apt-get install -y libssl-dev libcurl4-openssl-dev  # Needed for pyCurl
RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev
RUN apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
RUN apt-get install -y qtbase5-dev
RUN apt-get install -y qtdeclarative5-dev
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y wget

RUN apt-get install -y libssl-dev libcurl4-openssl-dev
RUN apt-get update && apt-get install -y python3.8-dev python3-pip
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN python -m pip install --upgrade pip

RUN pip install pytest pytest-forked lz4 pyyaml qt5-py
RUN pip install gym3
RUN pip install gym==0.21.0
RUN pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
RUN pip install kornia opencv-python
RUN pip install tensorflow tensorboard
RUN git clone https://github.com/openai/baselines.git
RUN cd baselines && pip install -e .

RUN pip install jaynes==0.8.11 ml_logger==v0.8.69 waterbear params-proto==2.9.6 functional-notations --no-cache-dir
# ADD mjkey.txt /root/.mujoco/mjkey.txt
RUN pip install google-cloud-storage
RUN pip install networkx
RUN pip install --upgrade gym==0.21.0

RUN yes | pip install flax
RUN yes | pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN python -m pip install git+https://github.com/adam-coogan/jaxinterp2d.git@master
RUN yes | pip install optax
RUN yes | pip install tensorflow_probability
RUN yes | pip install lenstronomy
RUN yes | conda install numba
RUN yes | pip install lenstronomy
RUN yes | conda install grpcio
RUN yes | pip install objax
RUN yes | cd lensing-minimal
RUN yes | conda install flax
RUN yes | pip install distrax
RUN yes | pip install numpyro

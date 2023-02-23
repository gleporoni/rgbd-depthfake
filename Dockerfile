FROM pytorchlightning/pytorch_lightning AS env

# Set the working directory
WORKDIR /workdir

# Install requirements
COPY requirements.txt .
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python
RUN pip install -r requirements.txt
ARG WANDB_API_KEY 
RUN wandb login $WANDB_API_KEY
RUN git config --global --add safe.directory /workdir

FROM env

# Install the package in the current directory
COPY setup.py src/ ./
RUN pip install -e .

# Copy the files
COPY . .
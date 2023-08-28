# NPCAI Llama.cpp Benchmark

This repository contains a benchmark script for [llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) API. Please refer to this document for how to install a Llama model and run the benchmark script against it. This assumes installation on a Linux Debian-based distribution (like Ubuntu).

# Setup Script Instructions

## Step 1: Create the setup script

Run the following command:

```
sudo vim setup.sh
```

Paste the following script:

```
#!/bin/bash

# Step 1: Update apt repository and upgrade packages
sudo apt-get update
sudo apt-get -y upgrade

# Step 2: Install CUDA keyring and CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
export PATH=/usr/local/cuda/bin:$PATH
source ~/.bashrc

# Step 3: Install python3-pip & git-lfs
sudo apt-get install -y python3-pip git-lfs ninja-build

#Step 4: Clone and build llama.cpp
git clone --progress --verbose https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make LLAMA_CUBLAS=1

# Step 5: Clone & configure llama.cpp repo
cd ..
git clone --progress --verbose https://github.com/NPCAI-Studio/llama-cpp-benchmark.git
cd llama-cpp-benchmark
pip install -r requirements.txt

# Step 6: Install a GGML model from HuggingFace
mkdir models
cd models
git lfs install
git clone --progress --verbose https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML

# Step 7: Convert GGML model to GGUF
cd ..
cd ..
cd llama.cpp
pip install -r requirements.txt
python3 convert-llama-ggmlv3-to-gguf.py --input "../llama-cpp-benchmark/models/llama-2-13b-chat.ggmlv3.q4_0.bin" --output "../llama-cpp-benchmark/models/gguf_model.bin" --name llama-2-13b-chat-gguf --desc "GGUF converted from llama-2-13b-chat.ggmlv3.q4_0.bin"

# Step 8: Run the benchmark
cd ..
cd llama-cpp-benchmark
echo ""
echo ""
echo "Setup completed succussfully. Run the benchmark with the following command:"
echo ""
echo "python3 npcai_benchmark.py --model_path models/gguf_model.bin --n_gpu_layers 48"
echo ""
echo ""
```

Save and close the file.

## Step 3: Make the file executable

Run the following command:

```
sudo chmod u+x setup.sh
```

## Step 4: Run the setup script

Run the following command:

```
sudo ./setup.sh
```

# Manual Setup Instructions

## Step 1: Install Python

Run the following commands:

```
sudo apt-get update
sudo apt-get install python3.10
sudo apt install python3-pip
```

## Step 2: Install git

Run the follow command:

```
sudo apt-get install git
```

## Step 3 (Optional):

This step is required if running this model with GPU offloading. This assumes the instance has NVIDIA GPUs running. Run the following commands.

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
sudo apt install -y ninja-build
export PATH=/usr/local/cuda/bin:$PATH
source ~/.bashrc
```

## Step 4: Install & Build Llama.cpp

Run the following commands:

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

If building to run solely on CPU, run this command:

```
make
```

If building to run with GPU offloading, instead run this command:

```
make LLAMA_CUBLAS=1
```

## Step 5: Install git-lfs

Run the following command:

```
sudo apt-get install git-lfs
```

## Step 6: Clone and configure this repository

Run the following commands (assuming the cwd is `llama.cpp` from step #3):

```
cd ..
git clone https://github.com/NPCAI-Studio/llama-cpp-benchmark.git
cd llama-cpp-benchmark
pip3 install -r requirements.txt
```

## Step 7: Install a GGML model from HuggingFace

Find the binary you want to install on HuggingFace. Copy the link and run the following commands (assuming the cwd is `llama-cpp-benchmark` from step #5):

```
mkdir models
cd models
git lfs install
sudo apt install wget
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin
```

*Update 8.28:* As of this date, the latest commit of the llama.cpp library requires a GGUF model format instead of GGML. TheBloke on HuggingFace has not yet added GGUF model versions, but has included a conversion script. To convert a downloaded GGML model to GGUF, run the following commands (assuming you are in the directory from the previous step):

```
cd ..
cd ..
cd llama.cpp
pip3 install -r requirements.txt
python3 convert-llama-ggmlv3-to-gguf.py --input "../llama-cpp-benchmark/models/llama-2-13b-chat.ggmlv3.q4_0.bin" --output "../llama-cpp-benchmark/models/gguf_model.bin" --name <model_name> --desc <model_desc>
```

## Step 8: Run the benchmark

Navigate back to the `llama-cpp-benchmark` repository root directory. Run the following command:

```
python3 npcai_benchmark.py --model_path "models/<model_name>"
```

If you wish to offload layers to the GPU, run with the following commands:

```
python3 npcai_benchmark.py --model_path "models/<model_name>" --n_gpu_layers 48
```

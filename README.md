# NPCAI Llama.cpp Benchmark

This repository contains a benchmark script for [llama.cpp](https://github.com/ggerganov/llama.cpp) using the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) API. Please refer to this document for how to install a Llama model and run the benchmark script against it.

## Step 1: Install Python

See instructions for installing Python 3 on Linux [here](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install-linux.html).

## Step 2: Install git

Run the follow command:

```
sudo apt-get update
sudo apt-get install git
```

## Step 3: Install & Build Llama.cpp

Run the following commands:

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

## Step 4: Install git-lfs

See instructions for installing git-lfs [here](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md).

## Step 5: Clone and configure this repository

Run the following commands (assuming the cwd is `llama.cpp` from step #3):

```
cd ..
git clone github.com/NPCAI-Studio/llama-cpp-benchmark.git
cd llama-cpp-benchmark
pip3 install -r requirements.txt
```

## Step 6: Install a GGML model from HuggingFace

Find the binary you want to install on HuggingFace. Copy the link and run the following commands (assuming the cwd is `llama-cpp-benchmark` from step #5):

```
mkdir models
cd models
git lfs install
sudo apt install wget
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin
```

## Step 7: Run the benchmark

Navigate back to the `llama-cpp-benchmark` repository root directory. Run the following command:

```
python3 npcai_benchmark.py --model_path “models/<model_name>”
```
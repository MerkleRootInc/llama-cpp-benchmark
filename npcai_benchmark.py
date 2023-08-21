import os, psutil, time, fire, subprocess, json, argparse

from llama_cpp import Llama, LlamaTokenizer
from typing import Optional

def main(
    model_path: str,
    repeat_penalty: float = 1.2,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 100,
    max_gen_len: int = 512,
):
    # Load the model
    llm = Llama(model_path=model_path)
    tokenizer = LlamaTokenizer(llama=llm)

    input_sequences = [
        "Who was George Washington?"
    ]

    print(f"Benchmarking inference for {len(input_sequences)} input sequences...\n")

    for sequence in input_sequences:
        tokens = tokenizer.encode(text=sequence, add_bos=False)
        start_time = time.time()
        gpu_percent = "N/A"
        vram_usage = "N/A"

        final_input = f"Respond to the question that begins after 'Q:'. The response should go after 'A:'. Q: {sequence} A:"
        output = llm(
            final_input,
            max_tokens=max_gen_len,
            echo=True,
            repeat_penalty=repeat_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        end_time = time.time()
        cpu_percent = psutil.cpu_percent(interval=None)
        if has_nvidia_gpu():
            gpu_percent = get_gpu_utilization()
            vram_usage = get_vram_usage()

        gen_text = output["choices"][0]["text"][len(final_input):]
        gen_tokens = tokenizer.encode(text=gen_text, add_bos=False)
        gen_speed = len(gen_tokens) / (end_time - start_time)

        print()
        print(f"Input sequence length: {len(tokens)}")
        print(f"Total inference time (sec.): {round(end_time - start_time, 2)}")
        print(f"Tokens generated: {len(gen_tokens)}")
        print(f"Inference-adjusted rate (tokens/sec.): {round(gen_speed, 2)}")
        print(f"CPU utilization (%): {cpu_percent}")
        print(f"GPU utilization (%): {gpu_percent}")
        print(f"vRAM usage (MB): {vram_usage}")

        print()
        print("Generated Text:")
        print(gen_text)

        print()
        print("=" * 40)

def has_nvidia_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if "NVIDIA-SMI" in result.stdout:
            return True
        return False
    except Exception as e:
        return False

def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print("Error:", e)
        return "N/A"

def get_vram_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return int(result.stdout.strip())
    except Exception as e:
        print("Error:", e)
        return "N/A"

if __name__ == "__main__":
    fire.Fire(main)
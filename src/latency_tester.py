import argparse
import time
import requests
import statistics
import tomli_w
import os
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# 測試用的長 Prompt
TEST_PROMPTS = [
    # 1. 技術演進分析 (模擬長文生成與推論)
    """
    You are a senior technology historian and computer architect.

    Context:
    The evolution of General Purpose GPU (GPGPU) computing has shifted significantly from fixed-function graphics pipelines in the early 2000s to the highly programmable, tensor-core accelerated architectures of today.
    Key milestones include the introduction of CUDA by NVIDIA in 2006, the development of OpenCL for cross-platform compatibility, and the rise of AMD's ROCm ecosystem.
    In the 2010s, the focus shifted towards deep learning, with hardware explicitly designed for matrix multiplications (e.g., Tensor Cores, Matrix Cores).
    Recently, the interconnects (NVLink, Infinity Fabric) and memory hierarchies (HBM, diverse cache levels) have become just as critical as the compute cores themselves to alleviate the memory wall bottleneck in Large Language Model training.

    Task:
    Based on the context above and your internal knowledge, generate a comprehensive report comparing the architectural philosophy differences between NVIDIA's "Hopper" architecture and AMD's "CDNA 3" architecture. Focus specifically on how each handles memory coherency and FP8 precision for LLM inference.
    """,

    # 2. 法律/條款摘要 (模擬 RAG 或文件分析 - 高 Prefill 負載)
    """
    You are a legal assistant specializing in software licensing and cloud service agreements.

    Document:
    [...Assume a long excerpt of a cloud Service Level Agreement (SLA) follows...]
    1. Service Availability: The Service Provider guarantees 99.9% uptime during any monthly billing cycle.
    2. Credits: In the event of downtime exceeding the allowance, customers are eligible for service credits equal to 10% of their monthly bill for every 1 hour of downtime, capped at 50% of the total monthly bill.
    3. Exclusions: Downtime caused by scheduled maintenance (with 24h notice), force majeure events, or customer-side network configurations is excluded from the uptime calculation.
    4. Termination: Customers may terminate the agreement for cause if the Service Availability drops below 95% for three consecutive months.
    5. Data Retention: Upon termination, customer data will be retained for 30 days before permanent deletion.
    [...Assume 20 more distinct clauses about liability, indemnity, and jurisdiction...]

    Task:
    Identify potential risks for a startup company relying on this service for a mission-critical real-time payment gateway. Specifically, analyze the "Exclusions" and "Termination" clauses and explain why the 99.9% guarantee might be insufficient for financial services.
    """,

    # 3. 程式碼重構 (模擬 Coding Assistant)
    """
    You are a senior software engineer specializing in Python and High-Performance Computing.

    Code Snippet:
    ```python
    def messy_matrix_processing(data, rows, cols):
        res = []
        for i in range(rows):
            row_list = []
            for j in range(cols):
                val = data[i][j]
                if val > 0:
                    temp = val * 2
                    if temp % 3 == 0:
                        temp = temp / 3
                    row_list.append(temp)
                else:
                    row_list.append(0)
            res.append(row_list)

        # ... verify results ...
        return res
    ```

    Task:
    The above code is functionally correct but computationally inefficient for large datasets.
    1. Rewrite this function using NumPy vectorization to eliminate the nested loops.
    2. Explain how the CPU cache locality is improved by your vectorized solution compared to the list-of-lists approach.
    3. Provide a brief example of how to parallelize this further using Numba `jit`.
    """,

    # 4. 創意寫作與世界觀建構 (模擬長文本生成)
    """
    You are a sci-fi novelist designing a setting for a cyberpunk story.

    Setting:
    The year is 2142. The city of Neo-Taipei is a multi-layered metropolis suspended above the rising sea levels.
    The "Upper District" utilizes atmospheric processors to keep the air clean, inhabited by the corporate elite who control the "Neuro-Link" network.
    The "Under-Tide" sector is submerged specifically during high tide, forcing its inhabitants to live in amphibious container homes.
    Currency has been replaced by "Compute-Credits", which are mined by human brain activity during sleep.

    Task:
    Write a prologue for a story following a protagonist named "Kai", a rogue technician who repairs obsolete servers in the Under-Tide. Start the scene with Kai discovering a pre-war AI fragment inside a water-damaged server rack during a typhoon. Describe the sensory details of the humidity, the sound of the storm, and the hum of the old hardware.
    """,

    # 5. 科學論文解釋 (模擬學術研究助手)
    """
    You are a research scientist explaining complex astrophysical simulations to a graduate student.

    Abstract:
    We present 'Galaxy-Sim', a new hybrid code coupling N-body dynamics for dark matter with mesh-based hydrodynamics for baryonic gas.
    Unlike traditional Smooth Particle Hydrodynamics (SPH), our approach uses an Adaptive Mesh Refinement (AMR) grid to capture shock waves with high fidelity.
    We apply this to simulate the merger of two neutron stars, focusing on the gravitational wave emission and the subsequent kilonova nucleosynthesis.
    Preliminary results suggest that magnetic field amplification at the merger interface is faster than previously predicted by pure magnetohydrodynamic (MHD) models.

    Task:
    Explain the key advantage of using AMR (Adaptive Mesh Refinement) over SPH (Smooth Particle Hydrodynamics) specifically in the context of "capturing shock waves" mentioned in the abstract. Use an analogy related to resolution scaling to make it clear.
    """
]

# Concatenate prompts to create a sufficiently long context for prefill testing
LONG_PROMPT = (TEST_PROMPTS[0] + TEST_PROMPTS[1] + TEST_PROMPTS[2] + TEST_PROMPTS[3] + TEST_PROMPTS[4]) * 20

def measure_request(client, model_name, prompt, max_tokens=20):
    """
    Sends a request to the model and measures:
    1. TTFT (Time To First Token)
    2. Total Latency
    3. TPS (Tokens Per Second)
    """
    start_time = time.time()
    ttft = 0
    first_token_time = 0
    token_count = 0

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            stream=True
        )

        for chunk in stream:
            # In vLLM streaming, each chunk typically corresponds to one token
            if chunk.choices[0].delta.content is not None:
                token_count += 1
                if first_token_time == 0:
                    first_token_time = time.time()
                    ttft = first_token_time - start_time

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate TPS (End-to-End)
        tps = token_count / total_time if total_time > 0 else 0

        return {
            "ttft": ttft,
            "total": total_time,
            "tps": tps,
            "tokens": token_count,
            "success": True
        }
    except Exception as e:
        print(f"Request failed: {e}")
        return {
            "ttft": 0,
            "total": 0,
            "tps": 0,
            "tokens": 0,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--producers", default="")
    parser.add_argument("--consumers", required=True)
    parser.add_argument("--output-dir", default=".", help="Directory to save the results")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    producer_urls = [u for u in args.producers.split(",") if u]
    consumer_urls = [u for u in args.consumers.split(",") if u]

    results = {
        "id": args.test_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "producer_count": len(producer_urls),
            "consumer_count": len(consumer_urls)
        },
        "metrics": {}
    }

    # Retrieve model name
    try:
        temp_client = OpenAI(base_url=consumer_urls[0], api_key="EMPTY")
        model_list = temp_client.models.list()
        model_name = model_list.data[0].id
    except:
        model_name = "unknown"

    print(f"\n--- 開始測試 ID: {args.test_id} ---")

    if producer_urls:
        print(">> Mode: Disaggregated (LMCache)")

        # 1. Producer Prefill
        p_client = OpenAI(base_url=producer_urls[0], api_key="EMPTY")
        print("Step 1: 請求 Producer (Prefill)...")
        p_res = measure_request(p_client, model_name, LONG_PROMPT)
        results["metrics"]["producer_prefill"] = p_res
        print(f"Producer TTFT: {p_res['ttft']:.4f}s | TPS: {p_res['tps']:.2f}")

        time.sleep(2)

        # 2. Consumer Decode
        c_results = []
        print("Step 2: 請求 Consumers (Cache Hit check)...")
        for i, c_url in enumerate(consumer_urls):
            c_client = OpenAI(base_url=c_url, api_key="EMPTY")
            print(f"  Testing Consumer {i} ({c_url})...")
            res = measure_request(c_client, model_name, LONG_PROMPT)
            c_results.append(res)
            print(f"  Consumer {i} TTFT: {res['ttft']:.4f}s | TPS: {res['tps']:.2f}")

        # Calculate averages for Consumers
        valid_results = [r for r in c_results if r['success']]
        if valid_results:
            avg_c_ttft = statistics.mean([r['ttft'] for r in valid_results])
            avg_c_tps = statistics.mean([r['tps'] for r in valid_results])

            results["metrics"]["consumer_avg_ttft"] = avg_c_ttft
            results["metrics"]["consumer_avg_tps"] = avg_c_tps

            if avg_c_ttft > 0 and p_res['ttft'] > 0:
                speedup = p_res['ttft'] / avg_c_ttft
                results["metrics"]["speedup_ratio"] = speedup
                print(f"🚀 Speedup Ratio: {speedup:.2f}x")

        results["metrics"]["consumers_detail"] = c_results

    else:
        print(">> Mode: Standalone (Baseline)")
        c_client = OpenAI(base_url=consumer_urls[0], api_key="EMPTY")
        print("Step 1: 請求 Standalone Node...")
        res = measure_request(c_client, model_name, LONG_PROMPT)
        results["metrics"]["baseline_run1"] = res
        print(f"Baseline TTFT: {res['ttft']:.4f}s | TPS: {res['tps']:.2f}")

        print("Step 2: 再次請求 (Check local cache)...")
        res2 = measure_request(c_client, model_name, LONG_PROMPT)
        results["metrics"]["baseline_run2"] = res2
        print(f"Baseline (Run2) TTFT: {res2['ttft']:.4f}s | TPS: {res2['tps']:.2f}")

    # Write results to file
    outfile = output_path / f"report_{args.test_id}.toml"
    with open(outfile, "wb") as f:
        tomli_w.dump(results, f)
    print(f"報告已儲存至 {outfile}")

if __name__ == "__main__":
    main()

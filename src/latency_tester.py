import argparse
import time
import requests
import statistics
import tomli_w
from datetime import datetime
from openai import OpenAI

# 測試用的 Prompt (足夠長以觸發 Cache 效果)
LONG_PROMPT = "You are an AI assistant. Please generate a very detailed report about the history of GPU architecture development from 2000 to 2025, focusing on parallel computing improvements." * 5

def measure_request(client, model_name, prompt, max_tokens=20):
    start_time = time.time()
    ttft = 0
    first_token_time = 0

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                if first_token_time == 0:
                    first_token_time = time.time()
                    ttft = first_token_time - start_time

        total_time = time.time() - start_time
        return {"ttft": ttft, "total": total_time, "success": True}
    except Exception as e:
        print(f"Request failed: {e}")
        return {"ttft": 0, "total": 0, "success": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--producers", default="")
    parser.add_argument("--consumers", required=True)
    args = parser.parse_args()

    # 解析 URLs
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

    # 取得模型名稱 (假設第一個 Consumer 可用)
    try:
        temp_client = OpenAI(base_url=consumer_urls[0], api_key="EMPTY")
        model_list = temp_client.models.list()
        model_name = model_list.data[0].id
    except:
        model_name = "unknown"

    print(f"\n--- 開始測試 ID: {args.test_id} ---")
    print(f"Producers: {len(producer_urls)}, Consumers: {len(consumer_urls)}")

    # 模式 A: Disaggregated (有 Producer 也有 Consumer)
    if producer_urls:
        print(">> Mode: Disaggregated (LMCache)")

        # 1. Hit Producer (Prefill + Store Cache)
        p_client = OpenAI(base_url=producer_urls[0], api_key="EMPTY") # 簡單起見只打第一個 Producer
        print("Step 1: 請求 Producer (Prefill)...")
        p_res = measure_request(p_client, model_name, LONG_PROMPT)
        results["metrics"]["producer_prefill"] = p_res
        print(f"Producer TTFT: {p_res['ttft']:.4f}s")

        time.sleep(2) # 等待傳輸

        # 2. Hit Consumer (Decode + Load Cache)
        # 我們測試「所有」Consumer 看看是否都能吃到 Cache
        c_results = []
        print("Step 2: 請求 Consumers (Cache Hit check)...")
        for i, c_url in enumerate(consumer_urls):
            c_client = OpenAI(base_url=c_url, api_key="EMPTY")
            print(f"  Testing Consumer {i} ({c_url})...")
            res = measure_request(c_client, model_name, LONG_PROMPT)
            c_results.append(res)
            print(f"  Consumer {i} TTFT: {res['ttft']:.4f}s")

        # 計算 Consumer 平均值
        avg_c_ttft = statistics.mean([r['ttft'] for r in c_results if r['success']])
        results["metrics"]["consumer_avg_ttft"] = avg_c_ttft
        results["metrics"]["consumers_detail"] = c_results

        # Speedup Ratio
        if avg_c_ttft > 0:
            speedup = p_res['ttft'] / avg_c_ttft
            results["metrics"]["speedup_ratio"] = speedup
            print(f"🚀 Speedup Ratio: {speedup:.2f}x")

    # 模式 B: Standalone / TP8 (只有 Consumer/Main node)
    else:
        print(">> Mode: Standalone (Baseline)")
        c_client = OpenAI(base_url=consumer_urls[0], api_key="EMPTY")
        print("Step 1: 請求 Standalone Node...")
        res = measure_request(c_client, model_name, LONG_PROMPT)
        results["metrics"]["baseline_run1"] = res
        print(f"Baseline TTFT: {res['ttft']:.4f}s")

        # 再跑一次看看 vLLM 內建的 Prefix Caching (如果有的話)
        print("Step 2: 再次請求 (Check local cache)...")
        res2 = measure_request(c_client, model_name, LONG_PROMPT)
        results["metrics"]["baseline_run2"] = res2
        print(f"Baseline (2nd run) TTFT: {res2['ttft']:.4f}s")

    # 儲存結果
    outfile = f"report_{args.test_id}.toml"
    with open(outfile, "wb") as f:
        tomli_w.dump(results, f)
    print(f"報告已儲存至 {outfile}")

if __name__ == "__main__":
    main()

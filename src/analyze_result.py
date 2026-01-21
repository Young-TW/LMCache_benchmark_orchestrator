import os
import tomli
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 設定路徑
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"

def load_reports():
    data = []

    # 遍歷 runs 目錄尋找所有 .toml 檔案
    for report_file in RUNS_DIR.glob("*/report_*.toml"):
        try:
            with open(report_file, "rb") as f:
                report = tomli.load(f)

            test_id = report.get("id", "unknown")
            metrics = report.get("metrics", {})
            config = report.get("config", {})

            # 解析測試組態 (例如從 ID 解析 1p7d, 2p6d)
            # 假設 ID 格式如: 1p7d_llama3_70b
            parts = test_id.split('_')
            topology = parts[0] if len(parts) > 0 else "unknown"
            model_tag = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

            # 提取數據
            p_ttft = metrics.get("producer_prefill", {}).get("ttft", 0)
            c_ttft = metrics.get("consumer_avg_ttft", 0)
            speedup = metrics.get("speedup_ratio", 0)

            # 只有當數據有效時才加入
            if p_ttft > 0 or c_ttft > 0:
                data.append({
                    "Test ID": test_id,
                    "Model": model_tag,
                    "Topology": topology, # 1p7d, 2p6d...
                    "Producer (P)": config.get("producer_count", 0),
                    "Consumer (C)": config.get("consumer_count", 0),
                    "Producer TTFT (s)": round(p_ttft, 4),
                    "Consumer TTFT (s)": round(c_ttft, 4),
                    "Speedup (x)": round(speedup, 2)
                })
        except Exception as e:
            print(f"⚠️ Error reading {report_file}: {e}")

    return pd.DataFrame(data)

def print_summary_table(df):
    if df.empty:
        print("沒有找到任何測試數據。")
        return

    # 排序：先按模型，再按拓撲
    df_sorted = df.sort_values(by=["Model", "Topology"])

    print("\n" + "="*80)
    print("📊 LMCache Benchmark Summary")
    print("="*80)
    print(df_sorted.to_markdown(index=False))
    print("="*80 + "\n")

def plot_charts(df):
    if df.empty:
        return

    # 設定繪圖風格
    sns.set_theme(style="whitegrid")

    # 找出有多少種模型
    models = df["Model"].unique()

    for model in models:
        model_df = df[df["Model"] == model].sort_values("Topology")

        # 準備繪圖數據 (Melt for seaborn)
        df_melted = model_df.melt(
            id_vars=["Topology"],
            value_vars=["Producer TTFT (s)", "Consumer TTFT (s)"],
            var_name="Role",
            value_name="Time (s)"
        )

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_melted, x="Topology", y="Time (s)", hue="Role", palette=["#e74c3c", "#2ecc71"])

        # 標註數值
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')

        plt.title(f"LMCache Latency Analysis: {model}")
        plt.ylabel("Time to First Token (seconds)")
        plt.xlabel("Topology Configuration")

        output_file = PROJECT_ROOT / "plots" / f"benchmark_{model}.png"
        plt.savefig(output_file)
        print(f"📈 圖表已儲存: {output_file}")
        plt.close()

if __name__ == "__main__":
    df = load_reports()
    print_summary_table(df)
    plot_charts(df)

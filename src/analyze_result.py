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
            parts = test_id.split('_')
            topology = parts[0] if len(parts) > 0 else "unknown"
            # 處理像 llama3_70b_tp1 這種多段的情況
            model_tag = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

            # 提取數據
            p_ttft = metrics.get("producer_prefill", {}).get("ttft", 0)
            c_ttft = metrics.get("consumer_avg_ttft", 0)
            speedup = metrics.get("speedup_ratio", 0)

            # 如果報告沒算 speedup 但有 ttft，我們自己算
            if speedup == 0 and c_ttft > 0 and p_ttft > 0:
                speedup = p_ttft / c_ttft

            if p_ttft > 0 or c_ttft > 0:
                data.append({
                    "Test ID": test_id,
                    "Model": model_tag,
                    "Topology": topology,
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

    df_sorted = df.sort_values(by=["Model", "Topology"])

    print("\n" + "="*80)
    print("📊 LMCache Benchmark Summary")
    print("="*80)
    # 若有安裝 tabulate 庫，to_markdown 會更好看
    try:
        print(df_sorted.to_markdown(index=False))
    except ImportError:
        print(df_sorted.to_string(index=False))
    print("="*80 + "\n")

def plot_charts(df):
    if df.empty:
        return

    # 設定全域繪圖風格
    sns.set_theme(style="whitegrid")

    models = df["Model"].unique()

    for model in models:
        # 依照拓撲排序 (例如 1p1d, 1p2d...)，這裡簡單用字串排序，若需特定順序可自定義
        model_df = df[df["Model"] == model].sort_values("Topology")

        # --- 圖表 1: TTFT 比較 (Producer vs Consumer) ---
        df_melted = model_df.melt(
            id_vars=["Topology"],
            value_vars=["Producer TTFT (s)", "Consumer TTFT (s)"],
            var_name="Role",
            value_name="Time (s)"
        )

        plt.figure(figsize=(12, 6))
        ax1 = sns.barplot(
            data=df_melted,
            x="Topology",
            y="Time (s)",
            hue="Role",
            palette=["#e74c3c", "#2ecc71"] # 紅色代表耗時(Producer), 綠色代表快速(Consumer)
        )

        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f')

        plt.title(f"LMCache Latency Analysis: {model}")
        plt.ylabel("Time to First Token (seconds)")
        plt.xlabel("Topology Configuration")
        plt.tight_layout()

        output_file_ttft = PROJECT_ROOT / "plots" / f"benchmark_ttft_{model}.png"
        plt.savefig(output_file_ttft)
        print(f"📈 TTFT 圖表已儲存: {output_file_ttft}")
        plt.close()

        # --- 圖表 2: Speedup Ratio (加速比) ---
        plt.figure(figsize=(12, 6))

        # 使用漸層色，加速越快顏色越深
        ax2 = sns.barplot(
            data=model_df,
            x="Topology",
            y="Speedup (x)",
            hue="Topology", # 根據拓撲上色
            palette="viridis",
            legend=False
        )

        # 加上數值標籤
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.2fx', padding=3)

        # 加一條 1x 的基準線 (雖然 LMCache 肯定大於 1)
        plt.axhline(1, color='red', linestyle='--', linewidth=1, label="Baseline (1x)")

        plt.title(f"LMCache Speedup Ratio: {model}")
        plt.ylabel("Speedup Factor (Higher is Better)")
        plt.xlabel("Topology Configuration")
        plt.tight_layout()

        output_file_speedup = PROJECT_ROOT / "plots" / f"benchmark_speedup_{model}.png"
        plt.savefig(output_file_speedup)
        print(f"🚀 Speedup 圖表已儲存: {output_file_speedup}")
        plt.close()

if __name__ == "__main__":
    df = load_reports()
    print_summary_table(df)
    plot_charts(df)
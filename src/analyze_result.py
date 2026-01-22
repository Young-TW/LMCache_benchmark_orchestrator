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
PLOTS_DIR = PROJECT_ROOT / "plots"

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

            # 解析測試組態
            parts = test_id.split('_')
            topology = parts[0] if len(parts) > 0 else "unknown"
            model_tag = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

            # 提取數據: TTFT
            p_ttft = metrics.get("producer_prefill", {}).get("ttft", 0)
            c_ttft = metrics.get("consumer_avg_ttft", 0)

            # 提取數據: TPS (Tokens Per Second)
            p_tps = metrics.get("producer_prefill", {}).get("tps", 0)
            c_tps = metrics.get("consumer_avg_tps", 0)

            # 提取數據: Speedup
            speedup = metrics.get("speedup_ratio", 0)

            # 自動補算 Speedup
            if speedup == 0 and c_ttft > 0 and p_ttft > 0:
                speedup = p_ttft / c_ttft

            if p_ttft > 0 or c_ttft > 0:
                data.append({
                    "Test ID": test_id,
                    "Model": model_tag,
                    "Topology": topology,
                    "P_Count": config.get("producer_count", 0),
                    "C_Count": config.get("consumer_count", 0),
                    "Producer TTFT (s)": round(p_ttft, 4),
                    "Consumer TTFT (s)": round(c_ttft, 4),
                    "Producer TPS": round(p_tps, 2),
                    "Consumer TPS": round(c_tps, 2),
                    "Speedup (x)": round(speedup, 2)
                })
        except Exception as e:
            print(f"⚠️ Error reading {report_file}: {e}")

    return pd.DataFrame(data)

def print_summary_table(df):
    if df.empty:
        print("沒有找到任何測試數據。")
        return

    # 依照模型和拓撲排序
    df_sorted = df.sort_values(by=["Model", "Topology"])

    print("\n" + "="*100)
    print("📊 LMCache Benchmark Summary (Latency & Throughput)")
    print("="*100)
    try:
        # 挑選關鍵欄位顯示
        view_cols = ["Model", "Topology", "Producer TTFT (s)", "Consumer TTFT (s)", "Producer TPS", "Consumer TPS", "Speedup (x)"]
        print(df_sorted[view_cols].to_markdown(index=False))
    except ImportError:
        print(df_sorted.to_string(index=False))
    print("="*100 + "\n")

def plot_charts(df):
    if df.empty:
        return

    # 確保 plots 目錄存在
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 設定全域繪圖風格
    sns.set_theme(style="whitegrid")

    models = df["Model"].unique()

    for model in models:
        model_df = df[df["Model"] == model].sort_values("Topology")

        # --- 圖表 1: TTFT (延遲比較) ---
        df_ttft = model_df.melt(
            id_vars=["Topology"],
            value_vars=["Producer TTFT (s)", "Consumer TTFT (s)"],
            var_name="Role",
            value_name="Time (s)"
        )

        plt.figure(figsize=(12, 6))
        ax1 = sns.barplot(
            data=df_ttft, x="Topology", y="Time (s)", hue="Role",
            palette=["#e74c3c", "#2ecc71"] # 紅(慢) vs 綠(快)
        )
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f')

        plt.title(f"LMCache Latency (TTFT): {model}")
        plt.ylabel("Time to First Token (seconds) [Lower is Better]")
        plt.xlabel("Topology")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"benchmark_ttft_{model}.png")
        plt.close()

        # --- 圖表 2: TPS (吞吐量比較) ---
        df_tps = model_df.melt(
            id_vars=["Topology"],
            value_vars=["Producer TPS", "Consumer TPS"],
            var_name="Role",
            value_name="Tokens/sec"
        )

        plt.figure(figsize=(12, 6))
        ax2 = sns.barplot(
            data=df_tps, x="Topology", y="Tokens/sec", hue="Role",
            palette=["#f39c12", "#3498db"] # 橘(Producer) vs 藍(Consumer)
        )
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f')

        plt.title(f"LMCache Throughput (TPS): {model}")
        plt.ylabel("Tokens Per Second [Higher is Better]")
        plt.xlabel("Topology")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"benchmark_tps_{model}.png")
        plt.close()

        # --- 圖表 3: Speedup Ratio (加速倍率) ---
        plt.figure(figsize=(12, 6))
        ax3 = sns.barplot(
            data=model_df, x="Topology", y="Speedup (x)", hue="Topology",
            palette="viridis", legend=False
        )
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%.2fx', padding=3)

        plt.axhline(1, color='red', linestyle='--', linewidth=1, label="Baseline (1x)")
        plt.title(f"LMCache Prefill Speedup Ratio: {model}")
        plt.ylabel("Speedup Factor")
        plt.xlabel("Topology")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"benchmark_speedup_{model}.png")
        plt.close()

    print(f"📈 所有圖表已儲存至: {PLOTS_DIR}")

if __name__ == "__main__":
    df = load_reports()
    print_summary_table(df)
    plot_charts(df)
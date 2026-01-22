import os
import tomli
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np  # 用於計算平均值

# ================= 路徑與環境配置 =================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
PLOTS_DIR = PROJECT_ROOT / "plots"

def parse_test_id(test_id):
    """
    根據新規則解析 ID:
    1. 1n_{model_name} -> Type: kv_both, Topology: Baseline (TP8)
    2. {topology}_{model_name} -> Type: disaggregated, Topology: {topology}
    """
    if test_id.startswith("1n_"):
        test_type = "kv_both"
        topology = "Baseline (TP8)"
        model_name = test_id[3:] # 切掉 "1n_"
    else:
        test_type = "disaggregated"
        parts = test_id.split('_')
        topology = parts[0]
        if len(parts) > 1:
            model_name = "_".join(parts[1:])
        else:
            model_name = "unknown"

    return test_type, topology, model_name

def load_reports():
    data = []
    print(f"📂 正在從 {RUNS_DIR} 讀取報告...")

    found_count = 0
    for report_file in RUNS_DIR.glob("*/report_*.toml"):
        try:
            with open(report_file, "rb") as f:
                report = tomli.load(f)

            test_id = report.get("id", "unknown")
            metrics = report.get("metrics", {})
            config = report.get("config", {})

            test_type, topology, model_tag = parse_test_id(test_id)

            # 初始化變數
            p_ttft, c_ttft = 0, 0
            p_tps, c_tps = 0, 0
            speedup = 0

            # ================= 針對不同結構讀取數據 =================
            if test_type == "kv_both":
                # [修正] Baseline 結構是 baseline_run1, baseline_run2...
                # 我們需要手動蒐集並計算平均
                ttft_values = []
                tps_values = []

                for key, val in metrics.items():
                    if key.startswith("baseline_run") and isinstance(val, dict):
                        if "ttft" in val:
                            ttft_values.append(val["ttft"])
                        if "tps" in val:
                            tps_values.append(val["tps"])

                # 計算平均值
                if ttft_values:
                    c_ttft = sum(ttft_values) / len(ttft_values)
                if tps_values:
                    c_tps = sum(tps_values) / len(tps_values)

                # Baseline 沒有 Producer
                p_ttft = 0
                p_tps = 0
                speedup = 0

            else:
                # Disaggregated 結構 (原本的邏輯)
                p_ttft = metrics.get("producer_prefill", {}).get("ttft", 0)
                c_ttft = metrics.get("consumer_avg_ttft", 0)
                p_tps = metrics.get("producer_prefill", {}).get("tps", 0)
                c_tps = metrics.get("consumer_avg_tps", 0)
                speedup = metrics.get("speedup_ratio", 0)

                # 若無 speedup 欄位則手動補算
                if speedup == 0 and c_ttft > 0 and p_ttft > 0:
                    speedup = p_ttft / c_ttft

            # ================= 儲存有效數據 =================
            if c_ttft > 0:
                found_count += 1
                data.append({
                    "Test ID": test_id,
                    "Model": model_tag,
                    "Type": test_type,
                    "Topology": topology,
                    "P_Count": config.get("producer_count", 0),
                    "C_Count": config.get("consumer_count", 1),
                    "Producer TTFT (s)": round(p_ttft, 4),
                    "Consumer TTFT (s)": round(c_ttft, 4),
                    "Producer TPS": round(p_tps, 2),
                    "Consumer TPS": round(c_tps, 2),
                    "Speedup (x)": round(speedup, 2)
                })
            else:
                # 只有當確實讀不到數據時才報錯
                pass

        except Exception as e:
            print(f"❌ 讀取錯誤 {report_file}: {e}")

    print(f"✅ 成功讀取 {found_count} 筆測試報告。")
    return pd.DataFrame(data)

def print_summary_table(df):
    if df.empty:
        print("沒有找到任何測試數據。")
        return

    df_sorted = df.sort_values(by=["Model", "Type", "Topology"])

    print("\n" + "="*110)
    print("📊 LMCache Benchmark Summary")
    print("="*110)
    try:
        view_cols = ["Model", "Topology", "Producer TTFT (s)", "Consumer TTFT (s)", "Consumer TPS", "Speedup (x)"]
        print(df_sorted[view_cols].to_markdown(index=False))
    except ImportError:
        print(df_sorted.to_string(index=False))
    print("="*110 + "\n")

def plot_charts(df):
    if df.empty:
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    models = df["Model"].unique()

    for model in models:
        model_df = df[df["Model"] == model].sort_values("Topology")

        # --- 圖表 1: TTFT ---
        df_ttft = model_df.melt(
            id_vars=["Topology"],
            value_vars=["Producer TTFT (s)", "Consumer TTFT (s)"],
            var_name="Role",
            value_name="Time (s)"
        )
        df_ttft = df_ttft[df_ttft["Time (s)"] > 0]

        plt.figure(figsize=(12, 6))
        ax1 = sns.barplot(
            data=df_ttft, x="Topology", y="Time (s)", hue="Role",
            palette={"Producer TTFT (s)": "#e74c3c", "Consumer TTFT (s)": "#2ecc71"}
        )
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f')

        plt.title(f"LMCache Latency (TTFT): {model}")
        plt.ylabel("Time to First Token (s) [Lower is Better]")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"benchmark_ttft_{model}.png")
        plt.close()

        # --- 圖表 2: TPS ---
        df_tps = model_df.melt(
            id_vars=["Topology"],
            value_vars=["Producer TPS", "Consumer TPS"],
            var_name="Role",
            value_name="Tokens/sec"
        )
        df_tps = df_tps[df_tps["Tokens/sec"] > 0]

        plt.figure(figsize=(12, 6))
        ax2 = sns.barplot(
            data=df_tps, x="Topology", y="Tokens/sec", hue="Role",
            palette={"Producer TPS": "#f39c12", "Consumer TPS": "#3498db"}
        )
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f')

        plt.title(f"LMCache Throughput (TPS): {model}")
        plt.ylabel("Tokens Per Second [Higher is Better]")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"benchmark_tps_{model}.png")
        plt.close()

        # --- 圖表 3: Speedup (僅 Disaggregated) ---
        df_speedup = model_df[model_df["Speedup (x)"] > 0]

        if not df_speedup.empty:
            plt.figure(figsize=(10, 6))
            ax3 = sns.barplot(
                data=df_speedup, x="Topology", y="Speedup (x)", hue="Topology",
                palette="viridis", legend=False
            )
            for container in ax3.containers:
                ax3.bar_label(container, fmt='%.2fx', padding=3)

            plt.axhline(1, color='red', linestyle='--', linewidth=1, label="No Gain (1x)")
            plt.title(f"LMCache Prefill Speedup Ratio: {model}")
            plt.ylabel("Speedup Factor")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"benchmark_speedup_{model}.png")
            plt.close()

    print(f"📈 所有圖表已儲存至: {PLOTS_DIR}")

if __name__ == "__main__":
    df = load_reports()
    print_summary_table(df)
    plot_charts(df)

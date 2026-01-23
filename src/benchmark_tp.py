import os
import time
import subprocess
import shutil
import requests
import yaml
import argparse
from pathlib import Path
from copy import deepcopy
from test_matrix import TEST_MATRIX

# ================= 路徑與環境配置 =================

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = os.getenv("LLM_MODELS_DIR", "/home/young/models")
TESTER_SCRIPT = SRC_DIR / "latency_tester.py"

# LMCache Source Config
LMCACHE_REPO = "https://github.com/Young-TW/LMCache.git"
LMCACHE_COMMIT = "505cd45d494d976acaf8d26e5aa598f30a6ea790"
LMCACHE_SRC_DIR = PROJECT_ROOT / "LMCache_src"

print(f"專案根目錄: {PROJECT_ROOT}")
print(f"模型來源路徑: {MODELS_DIR}")
print(f"LMCache 原始碼: {LMCACHE_SRC_DIR}")

COMMON_ENV = {
    "HF_HOME": "/app/model",
    "PYTORCH_ROCM_ARCH": "gfx942",
    "TORCH_DONT_CHECK_COMPILER_ABI": "1",
    "CXX": "hipcc",
    "BUILD_WITH_HIP": "1",
    "LMCACHE_CONFIG_FILE": "/app/lmcache_config.yaml",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "PYTHONHASHSEED": "0",

    # [關鍵修正] 強制使用穩定的 V0 Engine，避開 V1 的初始化問題
    "VLLM_USE_V1": "0",

    # [優化] 確保 NCCL 使用 P2P (MI300X 支援 xGMI)
    "NCCL_P2P_DISABLE": "0",
    # 如果沒有 Infiniband，可以加上這個避免偵測超時，但 MI300 機台通常有網卡
    # "NCCL_IB_DISABLE": "1",
}

def prepare_lmcache_source():
    if not LMCACHE_SRC_DIR.exists():
        print(f"📥 Cloning LMCache from {LMCACHE_REPO}...")
        subprocess.run(["git", "clone", LMCACHE_REPO, str(LMCACHE_SRC_DIR)], check=True)

    print(f"🔄 Checking out commit {LMCACHE_COMMIT}...")
    subprocess.run(["git", "fetch", "--all"], cwd=LMCACHE_SRC_DIR, check=True)
    subprocess.run(["git", "checkout", LMCACHE_COMMIT], cwd=LMCACHE_SRC_DIR, check=True)
    print("✅ Source code prepared.")

def generate_docker_compose(config, work_dir):
    services = {}
    port_map = {}

    full_model_path = Path(MODELS_DIR) / config["model_rel_path"]
    if not full_model_path.exists():
        print(f"⚠️ 警告: 模型路徑不存在: {full_model_path}")

    # Redis
    # [修改] kv_both 模式也可能需要 Redis (取決於 lmcache_config，這裡統一啟動)
    if config["type"] in ["disaggregated", "kv_both"]:
        services["redis"] = {
            "image": "bitnamilegacy/redis:7.4.2-debian-12-r6",
            "container_name": f"lmcache_redis_{config['id']}",
            "network_mode": "host",
            "command": 'redis-server --save "" --appendonly no'
        }

    vllm_template = {
        "image": "rocm/vllm-dev:nightly_main_20260112",
        "network_mode": "host",
        "ipc": "host",
        "shm_size": "64gb", # TP8 建議保持較大的 SHM
        "group_add": ["video"],
        "cap_add": ["SYS_PTRACE", "IPC_LOCK"],
        "security_opt": ["seccomp:unconfined"],
        "devices": ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"],
        "volumes": [
            f"{full_model_path}:/app/model",
            "./lmcache_config.yaml:/app/lmcache_config.yaml",
            f"/dev/shm/lmcache_{config['id']}:/dev/shm/lmcache_store",
            f"{LMCACHE_SRC_DIR}:/app/LMCache_src"
        ],
        "environment": deepcopy(COMMON_ENV)
    }

    current_gpu_idx = config["gpu_offset"]
    base_port = 8000

    def get_kv_config(role):
        return f'\\"kv_connector\\":\\"LMCacheConnectorV1\\", \\"kv_role\\":\\"{role}\\"'

    def build_command(role_json_content, port, tp_size):
        kv_json = "{" + role_json_content + "}"
        install_cmd = "python3 -m pip install --no-build-isolation -e /app/LMCache_src &&"

        vllm_cmd = f"""python3 -m vllm.entrypoints.openai.api_server
        --model /app/model --port {port} --tensor-parallel-size {tp_size}
        --max-model-len 32768 --kv-transfer-config "{kv_json}" --gpu-memory-utilization 0.95"""

        full_cmd = f"{install_cmd} {vllm_cmd}"
        return "bash -c '" + full_cmd.replace("\n", " ") + "'"

    # --- Producers ---
    if config["type"] == "disaggregated":
        p_count = config["producer_count"]
        p_tp = config["producer_tp"]

        for i in range(p_count):
            container_name = f"lmcache_{config['id']}_p{i}"
            s_name = f"producer_{i}"

            gpu_list = [str(x) for x in range(current_gpu_idx, current_gpu_idx + p_tp)]
            current_gpu_idx += p_tp

            svc = deepcopy(vllm_template)
            svc["container_name"] = container_name
            svc["environment"]["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)

            svc["command"] = build_command(get_kv_config("kv_producer"), base_port, p_tp)
            svc["depends_on"] = ["redis"]

            services[s_name] = svc
            port_map[base_port] = container_name
            base_port += 1

    # --- Consumers (or Combined/Standalone) ---
    c_count = config["consumer_count"]
    c_tp = config["consumer_tp"]

    for i in range(c_count):
        # 命名規則：disaggregated 用 c{i}, 其他用 standalone/kvboth
        suffix = f"c{i}" if config["type"] == "disaggregated" else f"kvboth_{i}"
        container_name = f"lmcache_{config['id']}_{suffix}"
        s_name = f"consumer_{i}" # 保持 consumer 名稱以便測試腳本識別

        gpu_list = [str(x) for x in range(current_gpu_idx, current_gpu_idx + c_tp)]
        current_gpu_idx += c_tp

        svc = deepcopy(vllm_template)
        svc["container_name"] = container_name
        svc["environment"]["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)

        # [修改] 根據類型決定啟動指令
        if config["type"] == "disaggregated":
            kv_role_config = get_kv_config("kv_consumer")
            svc["depends_on"] = ["redis"]
            svc["command"] = build_command(kv_role_config, base_port, c_tp)

        elif config["type"] == "kv_both":
            # [新增] kv_both 邏輯：安裝 LMCache 並設定角色為 kv_both
            kv_role_config = get_kv_config("kv_both")
            svc["depends_on"] = ["redis"]
            svc["command"] = build_command(kv_role_config, base_port, c_tp)

        else:
            # Baseline (Pure vLLM without LMCache)
            cmd = f"""python3 -m vllm.entrypoints.openai.api_server
                --model /app/model --port {base_port} --tensor-parallel-size {c_tp}
                --gpu-memory-utilization 0.95 --max-model-len 32768"""
            svc["command"] = "bash -c '" + cmd.replace("\n", " ") + "'"

        services[s_name] = svc
        port_map[base_port] = container_name
        base_port += 1

    # Output Files
    with open(work_dir / "docker-compose.yaml", "w") as f:
        yaml.dump({"version": "3.8", "services": services}, f)

    with open(CURRENT_FILE.parent / "config" / "lmcache_config.yaml", "r") as f:
        lmcache_template = f.read()
        with open(work_dir / "lmcache_config.yaml", "w") as f:
            f.write(lmcache_template)

    return port_map

def print_container_logs(container_name):
    print(f"\n🔴 [DEBUG] Dump Log for: {container_name}")
    print("=" * 60)
    try:
        subprocess.run(["docker", "logs", "--tail", "50", container_name], check=False)
    except Exception as e:
        print(f"無法取得 Log: {e}")
    print("=" * 60 + "\n")

def wait_for_services(port_map, timeout=900):
    ports = list(port_map.keys())
    print(f"⏳ 等待服務啟動 (Timeout: {timeout}s)... 目標 Ports: {ports}")

    start_time = time.time()
    pending_ports = set(ports)

    while pending_ports:
        if time.time() - start_time > timeout:
            print(f"\n❌ 啟動逾時！")
            for p in pending_ports:
                c_name = port_map.get(p, "unknown")
                print_container_logs(c_name)
            return False

        for port in list(pending_ports):
            try:
                requests.get(f"http://localhost:{port}/v1/models", timeout=2)
                print(f"✅ Port {port} 已就緒")
                pending_ports.remove(port)
            except:
                pass

        if pending_ports:
            time.sleep(10)
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0:
                print(f"   ...已等待 {elapsed}s，剩餘: {len(pending_ports)}")

    return True

def run_single_benchmark(config):
    test_id = config["id"]
    work_dir = RUNS_DIR / test_id

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================ 開始測試: {test_id} ================")

    port_map = generate_docker_compose(config, work_dir)

    try:
        print(f"🚀 啟動 Docker 環境 (Dir: {work_dir})...")
        subprocess.run(["docker", "compose", "up", "-d"], cwd=work_dir, check=True)

        if wait_for_services(port_map, timeout=900):
            sorted_ports = sorted(port_map.keys())

            p_count = config.get("producer_count", 0)

            if config["type"] == "disaggregated":
                p_urls = ",".join([f"http://localhost:{p}/v1" for p in sorted_ports[:p_count]])
                c_urls = ",".join([f"http://localhost:{p}/v1" for p in sorted_ports[p_count:]])
            else:
                p_urls = ""
                c_urls = ",".join([f"http://localhost:{p}/v1" for p in sorted_ports])

            cmd = [
                "uv", "run", str(TESTER_SCRIPT),
                "--test-id", test_id,
                "--producers", p_urls,
                "--consumers", c_urls,
                "--output-dir", str(work_dir)
            ]

            print(f"🧪 執行測試腳本...")
            subprocess.run(cmd, check=True)
        else:
            print("⚠️ 測試中止：服務啟動失敗")

    except Exception as e:
        print(f"❌ 發生未預期錯誤: {e}")
    finally:
        print(f"🧹 正在清理 {test_id}...")
        subprocess.run(["docker", "compose", "down"], cwd=work_dir)
        shm_path = Path(f"/dev/shm/lmcache_{test_id}")
        if shm_path.exists():
            shutil.rmtree(shm_path, ignore_errors=True)

if __name__ == "__main__":
    if not MODELS_DIR or not Path(MODELS_DIR).exists():
        print(f"❌ 錯誤：模型目錄 {MODELS_DIR} 不存在。")
        exit(1)

    prepare_lmcache_source()

    for config in TEST_MATRIX:
        run_single_benchmark(config)
        time.sleep(10)

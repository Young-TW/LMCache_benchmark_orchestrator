import os
import time
import subprocess
import shutil
import requests
import yaml
import argparse
from pathlib import Path
from copy import deepcopy

# ================= 路徑與環境配置 =================

# 1. 動態定位 src 目錄與專案根目錄
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent           # .../LMCache_benchmark_orchestrator/src
PROJECT_ROOT = SRC_DIR.parent           # .../LMCache_benchmark_orchestrator

# 2. 定義產出目錄 (所有生成的檔案都放在 runs 資料夾)
RUNS_DIR = PROJECT_ROOT / "runs"

# 3. 模型路徑 (優先讀取環境變數，否則使用預設值)
# 使用方法: export LLM_MODELS_DIR="/path/to/your/models"
MODELS_DIR = os.getenv("LLM_MODELS_DIR", "/home/young/models")

# 4. 測試腳本位置
TESTER_SCRIPT = SRC_DIR / "latency_tester.py"

print(f"專案根目錄: {PROJECT_ROOT}")
print(f"模型來源路徑: {MODELS_DIR}")
print(f"測試工作區: {RUNS_DIR}")

# ================= 測試矩陣 =================
# 在這裡定義您的各種組合
TEST_MATRIX = [
    {
        "id": "1p7d_llama3_70b",
        "model_rel_path": "Llama-3.3-70B-Instruct", # 相對於 MODELS_DIR 的路徑
        "type": "disaggregated",
        "producers": 1,
        "consumers": 7,
        "tp_per_instance": 1,
        "gpu_offset": 0
    },
    # 您可以在此加入更多組合 (如 2p6d, tp8_baseline 等)
]

# 通用容器環境變數
COMMON_ENV = {
    "HF_HOME": "/app/model",
    "PYTORCH_ROCM_ARCH": "gfx942",
    "TORCH_DONT_CHECK_COMPILER_ABI": "1",
    "CXX": "hipcc",
    "BUILD_WITH_HIP": "1",
    "LMCACHE_CONFIG_FILE": "/app/lmcache_config.yaml",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "PYTHONHASHSEED": "0"
}

def generate_docker_compose(config, work_dir):
    """
    動態生成 docker-compose.yaml
    work_dir: 該次測試的專屬目錄 (例如 runs/1p7d_llama3_70b)
    """

    services = {}
    full_model_path = Path(MODELS_DIR) / config["model_rel_path"]

    # 檢查模型路徑是否存在
    if not full_model_path.exists():
        print(f"⚠️ 警告: 模型路徑不存在: {full_model_path}")

    # LMCache Redis
    if config["type"] == "disaggregated":
        services["redis"] = {
            "image": "bitnamilegacy/redis:7.4.2-debian-12-r6",
            "container_name": f"lmcache_redis_{config['id']}",
            "network_mode": "host",
            "command": 'redis-server --save "" --appendonly no'
        }

    # vLLM Template
    vllm_template = {
        "image": "rocm/vllm-dev:nightly_main_20260112",
        "network_mode": "host",
        "group_add": ["video"],
        "cap_add": ["SYS_PTRACE"],
        "security_opt": ["seccomp:unconfined"],
        "devices": ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"],
        "volumes": [
            f"{full_model_path}:/app/model",
            "./lmcache_config.yaml:/app/lmcache_config.yaml",
            f"/dev/shm/lmcache_{config['id']}:/dev/shm/lmcache_store"
        ],
        "environment": deepcopy(COMMON_ENV)
    }

    current_gpu_idx = config["gpu_offset"]
    base_port = 8000

    # 建立 Producers
    if config["type"] == "disaggregated":
        for i in range(config["producers"]):
            s_name = f"producer_{i}"
            svc = deepcopy(vllm_template)
            svc["container_name"] = f"lmcache_{config['id']}_p{i}"

            gpus = ",".join([str(x) for x in range(current_gpu_idx, current_gpu_idx + config["tp_per_instance"])])
            current_gpu_idx += config["tp_per_instance"]

            svc["environment"]["CUDA_VISIBLE_DEVICES"] = gpus

            kv_config = '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_producer"}'
            cmd = f"""python3 -m vllm.entrypoints.openai.api_server
            --model /app/model
            --port {base_port}
            --tensor-parallel-size {config['tp_per_instance']}
            --max-model-len 8192
            --kv-transfer-config '{kv_config}'"""

            svc["command"] = "bash -c '" + cmd.replace("\n", " ") + "'"
            svc["depends_on"] = ["redis"]
            services[s_name] = svc
            base_port += 1

    # 建立 Consumers (或是 Standalone)
    num_consumers = config["consumers"]
    for i in range(num_consumers):
        s_name = f"consumer_{i}" if config["type"] == "disaggregated" else "vllm_standalone"
        svc = deepcopy(vllm_template)
        svc["container_name"] = f"lmcache_{config['id']}_c{i}"

        gpus = ",".join([str(x) for x in range(current_gpu_idx, current_gpu_idx + config["tp_per_instance"])])
        current_gpu_idx += config["tp_per_instance"]

        svc["environment"]["CUDA_VISIBLE_DEVICES"] = gpus

        kv_arg = ""
        if config["type"] == "disaggregated":
             kv_arg = "--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_consumer\"}'"
             svc["depends_on"] = ["redis"]

        cmd = f"""python3 -m vllm.entrypoints.openai.api_server
        --model /app/model
        --port {base_port}
        --tensor-parallel-size {config['tp_per_instance']}
        --max-model-len 8192
        {kv_arg}"""

        svc["command"] = "bash -c '" + cmd.replace("\n", " ") + "'"
        services[s_name] = svc
        base_port += 1

    # 寫入 docker-compose.yaml
    compose_data = {"version": "3.8", "services": services}
    with open(work_dir / "docker-compose.yaml", "w") as f:
        yaml.dump(compose_data, f)

    # 寫入 lmcache_config.yaml
    lmcache_conf = """
chunk_size: 256
local_device: "cpu"
remote_url: "redis://localhost:6379"
remote_serde: "cachegen"
    """
    with open(work_dir / "lmcache_config.yaml", "w") as f:
        f.write(lmcache_conf)

    return True

def wait_for_services(ports, timeout=900):
    """檢查所有 API 是否存活"""
    print(f"⏳ 等待服務啟動，目標 Ports: {ports}")
    start_time = time.time()
    pending_ports = set(ports)

    while pending_ports:
        if time.time() - start_time > timeout:
            print(f"❌ 逾時！無法啟動的 Ports: {pending_ports}")
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

    return True

def run_single_benchmark(config):
    test_id = config["id"]
    work_dir = RUNS_DIR / test_id

    # 清理並重建工作目錄
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================ 開始測試: {test_id} ================")
    generate_docker_compose(config, work_dir)

    try:
        # 啟動環境
        print(f"🚀 啟動 Docker 環境 (Dir: {work_dir})...")
        subprocess.run(["docker", "compose", "up", "-d"], cwd=work_dir, check=True)

        # 計算 Ports
        start_port = 8000
        producer_ports = []
        consumer_ports = []

        if config["type"] == "disaggregated":
            for _ in range(config["producers"]):
                producer_ports.append(start_port)
                start_port += 1
            for _ in range(config["consumers"]):
                consumer_ports.append(start_port)
                start_port += 1
        else:
            consumer_ports.append(start_port)
            start_port += 1

        all_ports = producer_ports + consumer_ports

        # 等待並執行測試
        if wait_for_services(all_ports):
            p_urls = ",".join([f"http://localhost:{p}/v1" for p in producer_ports])
            c_urls = ",".join([f"http://localhost:{p}/v1" for p in consumer_ports])

            # 呼叫測試腳本，並指定輸出目錄
            cmd = [
                "uv", "run", str(TESTER_SCRIPT),
                "--test-id", test_id,
                "--producers", p_urls,
                "--consumers", c_urls,
                "--output-dir", str(work_dir) # 將結果存在對應的工作目錄
            ]

            print(f"🧪 執行測試腳本...")
            subprocess.run(cmd, check=True)
        else:
            print("⚠️ 測試跳過：服務啟動失敗")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
    finally:
        # 清理環境
        print(f"🧹 正在清理 {test_id}...")
        subprocess.run(["docker", "compose", "down"], cwd=work_dir)

        # 清理 SHM (重要)
        shm_path = Path(f"/dev/shm/lmcache_{test_id}")
        if shm_path.exists():
            shutil.rmtree(shm_path, ignore_errors=True)

if __name__ == "__main__":
    if not MODELS_DIR or not Path(MODELS_DIR).exists():
        print(f"❌ 錯誤：模型目錄 {MODELS_DIR} 不存在。")
        print("請設定環境變數: export LLM_MODELS_DIR='/path/to/models'")
        exit(1)

    for config in TEST_MATRIX:
        run_single_benchmark(config)
        time.sleep(5)
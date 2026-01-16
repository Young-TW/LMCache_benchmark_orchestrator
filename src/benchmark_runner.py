import os
import time
import subprocess
import shutil
import requests
import yaml # pip install pyyaml
from pathlib import Path
from copy import deepcopy

# ================= 配置區域 =================
BASE_DIR = Path.home() / "lmcache_docker"
MODELS_DIR = "/home/young/models" # 宿主機模型路徑

# 定義測試矩陣：在此處新增各種組合
TEST_MATRIX = [
    # 案例 A: 1 Producer (GPU0), 7 Consumers (GPU1-7)
    {
        "id": "1p7d_llama3_70b",
        "model_path": "/app/model/Llama-3.3-70B-Instruct",
        "type": "disaggregated", # 分離式架構
        "producers": 1,
        "consumers": 7,
        "tp_per_instance": 1,    # 每個實例用幾張卡 (70B可能需要量化版才能跑TP1)
        "gpu_offset": 0          # 從第幾號 GPU 開始
    },
    # 案例 B: 2 Producers (GPU0-1), 6 Consumers (GPU2-7)
    {
        "id": "2p6d_llama3_70b",
        "model_path": "/app/model/Llama-3.3-70B-Instruct",
        "type": "disaggregated",
        "producers": 2,
        "consumers": 6,
        "tp_per_instance": 1,
        "gpu_offset": 0
    },
    # 案例 C: 4 Producers, 4 Consumers
    {
        "id": "4p4d_llama3_70b",
        "model_path": "/app/model/Llama-3.3-70B-Instruct",
        "type": "disaggregated",
        "producers": 4,
        "consumers": 4,
        "tp_per_instance": 1,
        "gpu_offset": 0
    },
    # 案例 D: 傳統 TP8 (8卡跑一個大模型，無 LMCache) - 作為對照組
    {
        "id": "tp8_baseline",
        "model_path": "/app/model/Llama-3.3-70B-Instruct",
        "type": "standalone",    # 單體架構
        "producers": 0,          # 不適用
        "consumers": 1,          # 1個大實例
        "tp_per_instance": 8,    # 佔用8張卡
        "gpu_offset": 0
    }
]

# 通用環境變數
COMMON_ENV = {
    "HF_HOME": "/app/model",
    "PYTORCH_ROCM_ARCH": "gfx942", # 依照您的 MI300/其他卡調整
    "TORCH_DONT_CHECK_COMPILER_ABI": "1",
    "CXX": "hipcc",
    "BUILD_WITH_HIP": "1",
    "LMCACHE_CONFIG_FILE": "/app/lmcache_config.yaml",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "PYTHONHASHSEED": "0"
}

# ================= 程式邏輯 =================

def generate_docker_compose(config, work_dir):
    """根據配置動態生成 docker-compose.yaml"""

    services = {}

    # 1. Redis (如果是 LMCache 模式)
    if config["type"] == "disaggregated":
        services["redis"] = {
            "image": "bitnamilegacy/redis:7.4.2-debian-12-r6",
            "container_name": f"lmcache_redis_{config['id']}",
            "network_mode": "host",
            "command": 'redis-server --save "" --appendonly no'
        }

    # 基礎 vLLM 設定樣板
    vllm_template = {
        "image": "rocm/vllm-dev:nightly_main_20260112",
        "network_mode": "host",
        "group_add": ["video"],
        "cap_add": ["SYS_PTRACE"],
        "security_opt": ["seccomp:unconfined"],
        "devices": ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"],
        "volumes": [
            f"{MODELS_DIR}:/app/model",
            "./lmcache_config.yaml:/app/lmcache_config.yaml",
            # 為每個測試 ID 建立獨立的 SHM，避免衝突
            f"/dev/shm/lmcache_{config['id']}:/dev/shm/lmcache_store"
        ],
        "environment": deepcopy(COMMON_ENV)
    }

    current_gpu_idx = config["gpu_offset"]
    base_port = 8000

    # 生成 Producer 列表
    if config["type"] == "disaggregated":
        for i in range(config["producers"]):
            s_name = f"producer_{i}"
            svc = deepcopy(vllm_template)
            svc["container_name"] = f"lmcache_{config['id']}_p{i}"

            # 計算 GPU ID (支援 TP > 1)
            gpus = ",".join([str(x) for x in range(current_gpu_idx, current_gpu_idx + config["tp_per_instance"])])
            current_gpu_idx += config["tp_per_instance"]

            svc["environment"]["CUDA_VISIBLE_DEVICES"] = gpus

            # 組裝 Command
            kv_config = '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_producer"}'
            cmd = f"""python3 -m vllm.entrypoints.openai.api_server
            --model {config['model_path']}
            --port {base_port}
            --tensor-parallel-size {config['tp_per_instance']}
            --max-model-len 8192
            --kv-transfer-config '{kv_config}'"""

            svc["command"] = "bash -c '" + cmd.replace("\n", " ") + "'"
            svc["depends_on"] = ["redis"]
            services[s_name] = svc
            base_port += 1

    # 生成 Consumer 列表 (或是 TP8 的單一實例)
    num_consumers = config["consumers"]
    for i in range(num_consumers):
        s_name = f"consumer_{i}" if config["type"] == "disaggregated" else "vllm_standalone"
        svc = deepcopy(vllm_template)
        svc["container_name"] = f"lmcache_{config['id']}_c{i}"

        gpus = ",".join([str(x) for x in range(current_gpu_idx, current_gpu_idx + config["tp_per_instance"])])
        current_gpu_idx += config["tp_per_instance"]

        svc["environment"]["CUDA_VISIBLE_DEVICES"] = gpus

        # 組裝 Command
        kv_arg = ""
        if config["type"] == "disaggregated":
             kv_arg = "--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_consumer\"}'"
             svc["depends_on"] = ["redis"]

        cmd = f"""python3 -m vllm.entrypoints.openai.api_server
        --model {config['model_path']}
        --port {base_port}
        --tensor-parallel-size {config['tp_per_instance']}
        --max-model-len 8192
        {kv_arg}"""

        svc["command"] = "bash -c '" + cmd.replace("\n", " ") + "'"
        services[s_name] = svc
        base_port += 1

    compose_data = {
        "version": "3.8",
        "services": services
    }

    with open(work_dir / "docker-compose.yaml", "w") as f:
        yaml.dump(compose_data, f)

    # 寫入 LMCache config
    lmcache_conf = """
chunk_size: 256
local_device: "cpu"
remote_url: "redis://localhost:6379"
remote_serde: "cachegen"
    """
    with open(work_dir / "lmcache_config.yaml", "w") as f:
        f.write(lmcache_conf)

    return base_port # 返回最後使用的 port 的下一個，或者用來計算總數

def wait_for_services(ports, timeout=900):
    """輪詢所有預期的 Port 直到全部 HTTP 200"""
    print(f"等待服務啟動，目標 Ports: {ports}")
    start_time = time.time()

    pending_ports = set(ports)

    while pending_ports:
        if time.time() - start_time > timeout:
            print(f"❌ 逾時！無法啟動的 Ports: {pending_ports}")
            return False

        for port in list(pending_ports):
            try:
                # 簡單檢查 health
                url = f"http://localhost:{port}/v1/models"
                requests.get(url, timeout=2)
                print(f"✅ Port {port} 已就緒")
                pending_ports.remove(port)
            except:
                pass

        if pending_ports:
            time.sleep(10)
            print(f"尚在等待: {pending_ports} ({int(time.time() - start_time)}s)")

    return True

def run_single_benchmark(config):
    test_id = config["id"]
    work_dir = BASE_DIR / test_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    print(f"\n================ 開始測試: {test_id} ================")
    print(f"配置: {config['type']}, P:{config.get('producers')}, C:{config['consumers']}, TP:{config['tp_per_instance']}")

    generate_docker_compose(config, work_dir)

    try:
        # 1. 啟動容器
        subprocess.run(["docker", "compose", "up", "-d"], cwd=work_dir, check=True)

        # 2. 計算預期的 Port 列表
        # 邏輯必須跟 generate_docker_compose 一致
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
            # Standalone mode
            consumer_ports.append(start_port) # 只有一個服務，視為 consumer
            start_port += 1

        all_ports = producer_ports + consumer_ports

        # 3. 等待就緒
        if wait_for_services(all_ports):
            # 4. 呼叫測試腳本
            # 組裝參數傳給測試腳本
            p_urls = ",".join([f"http://localhost:{p}/v1" for p in producer_ports])
            c_urls = ",".join([f"http://localhost:{p}/v1" for p in consumer_ports])

            cmd = ["uv", "run", "latency_tester.py",
                   "--test-id", test_id,
                   "--producers", p_urls,
                   "--consumers", c_urls]

            print(f"執行測試腳本: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=BASE_DIR / "tests", check=True)
        else:
            print("測試跳過：服務啟動失敗")

    except Exception as e:
        print(f"發生錯誤: {e}")
    finally:
        print(f"正在清理 {test_id}...")
        subprocess.run(["docker", "compose", "down"], cwd=work_dir)
        # 額外清理：確保 Redis 資料不殘留 (雖然 docker down 會移除容器，但 shm 要小心)
        # 我們的 volume 是掛載到 host 的 /dev/shm/lmcache_{id}，docker down 不會刪除它，手動刪除比較乾淨
        shm_path = Path(f"/dev/shm/lmcache_{test_id}")
        if shm_path.exists():
            shutil.rmtree(shm_path, ignore_errors=True)

if __name__ == "__main__":
    # 確保 uv 有安裝，或改用 python
    for config in TEST_MATRIX:
        run_single_benchmark(config)
        time.sleep(5) # 緩衝時間讓 GPU 記憶體釋放完全

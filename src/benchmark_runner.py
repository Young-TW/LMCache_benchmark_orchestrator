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

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = os.getenv("LLM_MODELS_DIR", "/home/young/models")
TESTER_SCRIPT = SRC_DIR / "latency_tester.py"

print(f"專案根目錄: {PROJECT_ROOT}")
print(f"模型來源路徑: {MODELS_DIR}")

# ================= 測試矩陣 =================
TEST_MATRIX = [
    {
        "id": "1p7d_llama3_70b",
        "model_rel_path": "Llama-3.3-70B-Instruct",
        "type": "disaggregated",
        "producers": 1,
        "consumers": 7,
        "tp_per_instance": 1,
        "gpu_offset": 0
    },
]

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
    生成配置並返回 port_map 以供後續追蹤
    """
    services = {}
    port_map = {}

    full_model_path = Path(MODELS_DIR) / config["model_rel_path"]
    if not full_model_path.exists():
        print(f"⚠️ 警告: 模型路徑不存在: {full_model_path}")

    # Redis
    if config["type"] == "disaggregated":
        services["redis"] = {
            "image": "bitnamilegacy/redis:7.4.2-debian-12-r6",
            "container_name": f"lmcache_redis_{config['id']}",
            "network_mode": "host",
            "command": 'redis-server --save "" --appendonly no'
        }

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

    def get_kv_config(role):
        return f'\\"kv_connector\\":\\"LMCacheConnectorV1\\", \\"kv_role\\":\\"{role}\\"'

    # Producers
    if config["type"] == "disaggregated":
        for i in range(config["producers"]):
            container_name = f"lmcache_{config['id']}_p{i}"
            s_name = f"producer_{i}"

            svc = deepcopy(vllm_template)
            svc["container_name"] = container_name

            gpus = ",".join([str(x) for x in range(current_gpu_idx, current_gpu_idx + config["tp_per_instance"])])
            current_gpu_idx += config["tp_per_instance"]

            svc["environment"]["CUDA_VISIBLE_DEVICES"] = gpus

            kv_json = "{" + get_kv_config("kv_producer") + "}"
            cmd = f"""python3 -m vllm.entrypoints.openai.api_server
            --model /app/model --port {base_port} --tensor-parallel-size {config['tp_per_instance']}
            --max-model-len 8192 --kv-transfer-config "{kv_json}" """

            svc["command"] = "bash -c '" + cmd.replace("\n", " ") + "'"
            svc["depends_on"] = ["redis"]

            services[s_name] = svc
            port_map[base_port] = container_name
            base_port += 1

    # Consumers
    num_consumers = config["consumers"]
    for i in range(num_consumers):
        container_name = f"lmcache_{config['id']}_c{i}" if config["type"] == "disaggregated" else "vllm_standalone"
        s_name = f"consumer_{i}" if config["type"] == "disaggregated" else "vllm_standalone"

        svc = deepcopy(vllm_template)
        svc["container_name"] = container_name

        gpus = ",".join([str(x) for x in range(current_gpu_idx, current_gpu_idx + config["tp_per_instance"])])
        current_gpu_idx += config["tp_per_instance"]

        svc["environment"]["CUDA_VISIBLE_DEVICES"] = gpus

        kv_arg = ""
        if config["type"] == "disaggregated":
             kv_json = "{" + get_kv_config("kv_consumer") + "}"
             kv_arg = f'--kv-transfer-config "{kv_json}"'
             svc["depends_on"] = ["redis"]

        cmd = f"""python3 -m vllm.entrypoints.openai.api_server
        --model /app/model --port {base_port} --tensor-parallel-size {config['tp_per_instance']}
        --max-model-len 8192 {kv_arg}"""

        svc["command"] = "bash -c '" + cmd.replace("\n", " ") + "'"
        services[s_name] = svc
        port_map[base_port] = container_name
        base_port += 1

    # Output Files
    with open(work_dir / "docker-compose.yaml", "w") as f:
        yaml.dump({"version": "3.8", "services": services}, f)

    lmcache_conf = """
chunk_size: 256
local_device: "cpu"
remote_url: "redis://localhost:6379"
remote_serde: "cachegen"
    """
    with open(work_dir / "lmcache_config.yaml", "w") as f:
        f.write(lmcache_conf)

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
            p_count = config.get("producers", 0)

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

    for config in TEST_MATRIX:
        run_single_benchmark(config)
        time.sleep(10)

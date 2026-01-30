# ================= 測試矩陣 =================
TEST_MATRIX = [
    # --- Baseline: TP kv_both ---
    {
        "id": "tp1_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "kv_both",
        "producer_count": 0,
        "producer_tp": 0,
        "consumer_count": 1,
        "consumer_tp": 1,
        "gpu_offset": 0
    },
    {
        "id": "tp2_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "kv_both",
        "producer_count": 0,
        "producer_tp": 0,
        "consumer_count": 1,
        "consumer_tp": 2,
        "gpu_offset": 0
    },
    {
        "id": "tp4_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "kv_both",
        "producer_count": 0,
        "producer_tp": 0,
        "consumer_count": 1,
        "consumer_tp": 4,
        "gpu_offset": 0
    },
    {
        "id": "tp8_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "kv_both",
        "producer_count": 0,
        "producer_tp": 0,
        "consumer_count": 1,
        "consumer_tp": 8,
        "gpu_offset": 0
    },
    # --- Producer TP = 1 ---
    {
        "id": "1p1d_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "disaggregated",
        "producer_count": 1, "producer_tp": 1,
        "consumer_count": 1, "consumer_tp": 1,
        "gpu_offset": 0
    },
    {
        "id": "1p2d_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "disaggregated",
        "producer_count": 1, "producer_tp": 1,
        "consumer_count": 1, "consumer_tp": 2,
        "gpu_offset": 0
    },
    {
        "id": "1p4d_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "disaggregated",
        "producer_count": 1, "producer_tp": 1,
        "consumer_count": 1, "consumer_tp": 4,
        "gpu_offset": 0
    },

    # --- Producer TP = 2 ---
    {
        "id": "2p1d_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "disaggregated",
        "producer_count": 1, "producer_tp": 2,
        "consumer_count": 1, "consumer_tp": 1,
        "gpu_offset": 0
    },
    {
        "id": "2p2d_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "disaggregated",
        "producer_count": 1, "producer_tp": 2,
        "consumer_count": 1, "consumer_tp": 2,
        "gpu_offset": 0
    },
    {
        "id": "2p4d_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "disaggregated",
        "producer_count": 1, "producer_tp": 2,
        "consumer_count": 1, "consumer_tp": 4,
        "gpu_offset": 0
    },

    # --- Producer TP = 4 ---
    {
        "id": "4p1d_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "disaggregated",
        "producer_count": 1, "producer_tp": 4,
        "consumer_count": 1, "consumer_tp": 1,
        "gpu_offset": 0
    },
    {
        "id": "4p2d_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "disaggregated",
        "producer_count": 1, "producer_tp": 4,
        "consumer_count": 1, "consumer_tp": 2,
        "gpu_offset": 0
    },
    {
        "id": "4p4d_gpt-oss-120b",
        "model_rel_path": "gpt-oss-120b",
        "type": "disaggregated",
        "producer_count": 1, "producer_tp": 4,
        "consumer_count": 1, "consumer_tp": 4,
        "gpu_offset": 0
    },
]

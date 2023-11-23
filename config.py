import os


class ServerConfig:
    """配置类"""
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    SERVER_CENTER_URL = "127.0.0.1"
    SERVER_CENTER_PORT = 8000
    # 最大超时时间
    SESSION_TIMEOUT = 600

    REGISTER_URL = "/model/register"
    HEARTBEAT_URL = "/model/heartbeat"
    KILL_SIGNAL_URL = "/v1/completions/kill"
    # 心跳频率，单位秒
    HEARTBEAT_RATE = 30
    # 最大心跳失败次数
    MAX_HEARTBEAT_FAILURES = 3

    # HUGGINGFACE_HUB_TOKEN = "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"
    HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Load模型时的默认访问地址
    # MODEL_CACHED_PATH = "/home/wangchen/llm/customllm/models/"
    MODEL_CACHED_PATH = "/mnt/disk1/wangchen/llm/models/"

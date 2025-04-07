
OPENAI_LIST = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4.5-preview",
               "/home/mnt/wdxu/models/DeepSeek-R1-Distill-Qwen-7B",
               "/home/mnt/wdxu/models/Qwen2.5-Coder-14B-Instruct",
               "/home/mnt/wdxu/models/Qwen2.5-Coder-32B-Instruct",
               "/app/wdxu/models/Qwen2.5-Coder-32B",
               "/app/wdxu/models/DeepSeek-R1-Distill-Qwen-32B",
               "glm-4-flash",
               "/app/wdxu/models/DeepSeek-R1-Distill-Qwen-7B"]

MODEL_LIMITS = {
    # "claude-instant-1": 100_000,
    # "claude-2": 100_000,
    # "claude-3-opus-20240229": 200_000,
    # "claude-3-sonnet-20240229": 200_000,
    # "claude-3-haiku-20240307": 200_000,
    "gpt-3.5-turbo": 16_385,
    "gpt-4": 8_192,
    "gpt-4o": 128_000,
    "gpt-4.5-preview": 128_000,
    "glm-4-flash": 32_768,
}
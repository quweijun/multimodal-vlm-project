"""
模型配置文件
"""

# LLaVA模型配置
LLAVA_CONFIG = {
    "vision_encoder": "openai/clip-vit-large-patch14-336",
    "language_model": "lmsys/vicuna-7b-v1.5",
    "projection_dim": 512,
    "image_size": 336,
    "vision_feature_layer": -2,
}

# Qwen-VL模型配置
QWEN_VL_CONFIG = {
    "model_name": "Qwen/Qwen3-VL-4B-Instruct",
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "trust_remote_code": True,
    "max_length": 2048,
}

# 训练配置
TRAINING_CONFIG = {
    "learning_rate": 5e-5,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "warmup_steps": 100,
    "logging_steps": 50,
    "save_steps": 500,
}

# LoRA配置
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
}

def get_model_config(model_type="qwen_vl"):
    """获取模型配置"""
    if model_type == "qwen_vl":
        return QWEN_VL_CONFIG
    elif model_type == "llava":
        return LLAVA_CONFIG
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def get_training_config():
    """获取训练配置"""
    return TRAINING_CONFIG.copy()

def get_lora_config():
    """获取LoRA配置"""
    return LORA_CONFIG.copy()
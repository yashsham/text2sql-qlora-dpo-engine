"""
train/config.py
Centralized configuration for QLoRA SFT and DPO training.
Supports full 7B/8B training on GPUs as well as dry-run execution.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    # Model Selection
    # Default to modern state-of-the-art coding SLM (Qwen 2.5 Coder 7B or Llama-3.1 8B)
    base_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    lightweight_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # For rapid local testing

    # Quantization (QLoRA 4-bit NF4)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # LoRA Architecture
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Stage 1: SFT Parameters
    sft_learning_rate: float = 2e-4
    sft_num_epochs: int = 3
    sft_batch_size: int = 2
    sft_gradient_accumulation_steps: int = 4
    sft_output_dir: str = "./outputs/sft_adapter"

    # Stage 2: DPO Parameters
    dpo_learning_rate: float = 5e-6
    dpo_beta: float = 0.1  # Regularization parameter controlling deviation from reference model
    dpo_num_epochs: int = 2
    dpo_batch_size: int = 1
    dpo_gradient_accumulation_steps: int = 8
    dpo_output_dir: str = "./outputs/dpo_adapter"

    # Sequence lengths
    max_seq_length: int = 2048
    max_prompt_length: int = 1024

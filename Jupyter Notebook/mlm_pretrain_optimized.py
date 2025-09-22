# mlm_pretrain_optimized.py (최종 버전)

import os
import math
import torch
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ====================================================================
# 설정 클래스 (기존과 동일)
# ====================================================================
@dataclass
class OptimizedMLMConfig:
    base_model_name: str = "beomi/KcELECTRA-base-v2022" # 이전 모델: klue/bert-base
    txt_file_path: str = "../NIKL_AU_2023_COMPETITION_v1.0/combined.txt"
    output_model_path: str = "../mlm_finetuned_model"
    block_size: int = 512
    epochs: int = 3
    train_batch_size: int = 4
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = False # default: True. klue/bert-base 시에는 활성화.
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    preprocessing_num_proc: int = 16
    dataset_cache_file_name: str = "cached_dataset"
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    mlm_probability: float = 0.15
    validation_split: float = 0.01


# ====================================================================


def run_optimized_mlm_pretraining(config: OptimizedMLMConfig):
    """최적화된 MLM 사전학습 함수 (기본 Trainer 사용)"""

    logger.info(f"GPU 사용 가능: {torch.cuda.is_available()}")

    # 캐시 파일 경로 설정
    cache_dir = os.path.join(os.path.dirname(config.txt_file_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 데이터셋 로딩 및 전처리
    raw_dataset = load_dataset(
        "text", data_files=config.txt_file_path, split="train", cache_dir=cache_dir
    )
    dataset_split = raw_dataset.train_test_split(
        test_size=config.validation_split, seed=42
    )
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = AutoModelForMaskedLM.from_pretrained(config.base_model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.block_size,
            padding=False,
        )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= config.block_size:
            total_length = (total_length // config.block_size) * config.block_size
        result = {
            k: [
                t[i : i + config.block_size]
                for i in range(0, total_length, config.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config.preprocessing_num_proc,
        remove_columns=train_dataset.column_names,
    ).map(group_texts, batched=True, num_proc=config.preprocessing_num_proc)
    lm_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config.preprocessing_num_proc,
        remove_columns=eval_dataset.column_names,
    ).map(group_texts, batched=True, num_proc=config.preprocessing_num_proc)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability
    )

    # [핵심] 학습 인수 설정 - 여기에 모든 최적화 설정을 담습니다.
    training_args = TrainingArguments(
        output_dir=config.output_model_path,
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # 옵티마이저와 스케줄러 설정을 직접 지정 (Trainer가 자동으로 생성)
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # [핵심] 기본 Trainer를 사용하여 안정성 확보
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 학습 시작
    trainer.train()

    # 최종 모델 저장
    final_model_path = os.path.join(config.output_model_path, "final")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"최종 모델이 '{final_model_path}'에 저장되었습니다.")


if __name__ == "__main__":
    config = OptimizedMLMConfig()
    run_optimized_mlm_pretraining(config)

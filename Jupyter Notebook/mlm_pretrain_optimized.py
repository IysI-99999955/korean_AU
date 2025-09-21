# mlm_pretrain_optimized.py

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
    get_linear_schedule_with_warmup,
)
from transformers.optimization import AdamW
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ====================================================================
# 최적화된 설정 클래스
# ====================================================================
@dataclass
class OptimizedMLMConfig:
    # --- 모델 및 데이터 설정 ---
    base_model_name: str = "klue/bert-base"
    txt_file_path: str = "../NIKL_AU_2023_COMPETITION_v1.0/combined.txt"
    output_model_path: str = "../mlm_finetuned_model"

    # --- 토크나이징 설정 (BERT 기준 최적화) ---
    block_size: int = 512  # BERT 최대 길이로 증가

    # --- 학습 하이퍼파라미터 (고성능 서버 최적화) ---
    epochs: int = 3  # 일반적으로 MLM은 3-5 에포크면 충분
    train_batch_size: int = 64  # GPU 메모리에 따라 조정 필요
    eval_batch_size: int = 128
    gradient_accumulation_steps: int = 2  # 효과적인 배치 크기 증가

    # --- 최적화 설정 ---
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # --- 성능 최적화 설정 ---
    fp16: bool = True  # 혼합 정밀도
    dataloader_num_workers: int = 8  # CPU 코어 수에 맞춰 조정
    dataloader_pin_memory: bool = True

    # --- 처리 최적화 ---
    preprocessing_num_proc: int = 16  # 32CPU의 절반 사용
    dataset_cache_file_name: str = "cached_dataset"

    # --- 저장 설정 ---
    save_steps: int = 5000
    eval_steps: int = 5000
    save_total_limit: int = 3

    # --- MLM 설정 ---
    mlm_probability: float = 0.15

    # --- 검증 데이터 비율 ---
    validation_split: float = 0.01  # 1%를 검증용으로 사용


# ====================================================================


def get_dataset_size(file_path: str) -> int:
    """파일 크기를 통해 대략적인 데이터 크기 추정"""
    try:
        file_size = os.path.getsize(file_path)
        # 대략적인 추정: 1MB당 약 200,000 토큰
        estimated_tokens = file_size * 200
        return estimated_tokens
    except:
        return 0


class OptimizedTrainer(Trainer):
    """성능 최적화된 커스텀 Trainer"""

    def create_optimizer(self):
        """AdamW 옵티마이저 생성"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=1e-8,
        )
        # 'NoneType' object has no attribute 'param_groups' 오류 해결책으로 아래 한줄 추가
        self.optimizer = optimizer
        return optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """학습률 스케줄러 생성"""
        if optimizer is None:
            optimizer = self.optimizer

        warmup_steps = int(num_training_steps * self.args.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler


def run_optimized_mlm_pretraining(config: OptimizedMLMConfig):
    """최적화된 MLM 사전학습 함수"""

    logger.info(f"GPU 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"사용 가능한 GPU 수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 데이터셋 로딩
    logger.info(f"'{config.txt_file_path}' 파일에서 데이터셋을 로딩합니다.")
    if not os.path.exists(config.txt_file_path):
        raise FileNotFoundError(
            f"오류: '{config.txt_file_path}' 파일을 찾을 수 없습니다."
        )

    # 데이터 크기 추정
    estimated_size = get_dataset_size(config.txt_file_path)
    logger.info(f"추정 데이터 크기: {estimated_size:,} 토큰")

    # 캐시 파일 경로 설정
    cache_dir = os.path.join(os.path.dirname(config.txt_file_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    raw_dataset = load_dataset(
        "text", data_files=config.txt_file_path, split="train", cache_dir=cache_dir
    )

    # 검증 데이터 분할
    if config.validation_split > 0:
        dataset_split = raw_dataset.train_test_split(
            test_size=config.validation_split, seed=42
        )
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        logger.info(
            f"훈련 데이터: {len(train_dataset):,}, 검증 데이터: {len(eval_dataset):,}"
        )
    else:
        train_dataset = raw_dataset
        eval_dataset = None
        logger.info(f"훈련 데이터: {len(train_dataset):,}")

    # 토크나이저 및 모델 로딩
    logger.info("토크나이저와 모델을 로딩합니다.")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = AutoModelForMaskedLM.from_pretrained(config.base_model_name)

    # 특수 토큰 확인
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        """효율적인 토크나이징 함수"""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.block_size,
            padding=False,
            return_special_tokens_mask=True,
        )

    def group_texts(examples):
        """텍스트를 고정 길이 청크로 그룹화"""
        # 모든 텍스트를 연결
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # 블록 크기로 나누어 떨어지도록 자르기
        if total_length >= config.block_size:
            total_length = (total_length // config.block_size) * config.block_size

        # 블록 단위로 분할
        result = {
            k: [
                t[i : i + config.block_size]
                for i in range(0, total_length, config.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # 데이터셋 전처리 (기존 캐시가 있으면 재사용)
    logger.info("데이터셋 토크나이징을 시작합니다... (기존 캐시가 있으면 재사용)")

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config.preprocessing_num_proc,
        remove_columns=train_dataset.column_names,
        cache_file_name=os.path.join(
            cache_dir, f"{config.dataset_cache_file_name}_train_tokenized.arrow"
        ),
        desc="토크나이징 훈련 데이터",
        load_from_cache_file=True,  # 캐시 파일이 있으면 로드
    )

    logger.info("텍스트 그룹화를 시작합니다...")
    lm_train_dataset = tokenized_train.map(
        group_texts,
        batched=True,
        num_proc=config.preprocessing_num_proc,
        cache_file_name=os.path.join(
            cache_dir, f"{config.dataset_cache_file_name}_train_grouped.arrow"
        ),
        desc="그룹화 훈련 데이터",
    )

    lm_eval_dataset = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=config.preprocessing_num_proc,
            remove_columns=eval_dataset.column_names,
            cache_file_name=os.path.join(
                cache_dir, f"{config.dataset_cache_file_name}_eval_tokenized.arrow"
            ),
            desc="토크나이징 검증 데이터",
        )

        lm_eval_dataset = tokenized_eval.map(
            group_texts,
            batched=True,
            num_proc=config.preprocessing_num_proc,
            cache_file_name=os.path.join(
                cache_dir, f"{config.dataset_cache_file_name}_eval_grouped.arrow"
            ),
            desc="그룹화 검증 데이터",
        )

    logger.info("데이터셋 전처리 완료.")
    logger.info(f"최종 훈련 샘플 수: {len(lm_train_dataset):,}")
    if lm_eval_dataset:
        logger.info(f"최종 검증 샘플 수: {len(lm_eval_dataset):,}")

    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
    )

    # 학습 인수 설정
    total_steps = (
        math.ceil(
            len(lm_train_dataset)
            / (config.train_batch_size * config.gradient_accumulation_steps)
        )
        * config.epochs
    )

    training_args = TrainingArguments(
        output_dir=config.output_model_path,
        overwrite_output_dir=True,
        # 학습 설정
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # 최적화 설정
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        # 저장 및 평가
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy="steps" if lm_eval_dataset else "no",
        eval_steps=config.eval_steps if lm_eval_dataset else None,
        # 성능 최적화
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        remove_unused_columns=False,
        # 로깅
        logging_steps=100,
        report_to="none",
        # 기타
        prediction_loss_only=True,
        load_best_model_at_end=True if lm_eval_dataset else False,
        metric_for_best_model="eval_loss" if lm_eval_dataset else None,
    )

    # 트레이너 설정
    trainer = OptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 학습 시작
    logger.info("=== MLM 사전학습 시작 ===")
    logger.info(f"총 스텝 수: {total_steps:,}")
    logger.info(f"워밍업 스텝: {int(total_steps * config.warmup_ratio):,}")

    try:
        trainer.train()
        logger.info("=== MLM 사전학습 완료 ===")

        # 최종 모델 저장
        final_model_path = os.path.join(config.output_model_path, "final")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        logger.info(f"최종 모델이 '{final_model_path}'에 저장되었습니다.")

        # 메모리 정리
        del trainer, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except KeyboardInterrupt:
        logger.info("학습이 중단되었습니다.")
        # 현재까지의 모델 저장
        checkpoint_path = os.path.join(
            config.output_model_path, "interrupted_checkpoint"
        )
        trainer.save_model(checkpoint_path)
        logger.info(f"중단된 모델이 '{checkpoint_path}'에 저장되었습니다.")


if __name__ == "__main__":
    # 최적화된 설정
    config = OptimizedMLMConfig(
        # 필요시 여기서 설정 수정
        epochs=3,
        train_batch_size=32,  # GPU 메모리에 따라 조정
        block_size=512,
        preprocessing_num_proc=16,  # 32CPU의 절반
        dataloader_num_workers=8,
    )

    # 학습 실행
    run_optimized_mlm_pretraining(config)

# mlm_pretrain.py

import os
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ====================================================================
# 사용자가 수정할 모든 설정을 Config 클래스로 관리
# ====================================================================
@dataclass
class MLMConfig:
    # --- 사전 학습을 진행할 기반 모델 이름 ---
    base_model_name: str = "klue/bert-base"

    # --- 라벨 없는 대용량 텍스트 파일 경로 ---
    txt_file_path: str = "../NIKL_AU_2023_COMPETITION_v1.0/combined.txt"

    # --- MLM 추가 학습 후 새로 저장될 모델 폴더 경로 ---
    output_model_path: str = "../mlm_finetuned_model"

    # --- 토크나이징 시 사용할 최대 길이 (Chunk size) ---
    block_size: int = 128

    # --- 학습 관련 하이퍼파라미터 ---
    epochs: int = 10
    batch_size: int = 32
    fp16: bool = True  # GPU 지원 시 혼합 정밀도 학습


# ====================================================================


def run_mlm_pretraining(config: MLMConfig):
    """Config 객체를 받아 MLM 추가 학습을 수행하는 메인 함수"""

    print(f"'{config.txt_file_path}' 파일에서 데이터셋을 로딩합니다.")
    if not os.path.exists(config.txt_file_path):
        raise FileNotFoundError(
            f"오류: '{config.txt_file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요."
        )

    raw_dataset = load_dataset("text", data_files=config.txt_file_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = AutoModelForMaskedLM.from_pretrained(config.base_model_name)

    def tokenize_and_chunk(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
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

    print("데이터셋 토크나이징 및 Chunking을 시작합니다...")
    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenizer(
            examples["text"], truncation=False
        ),  # Chunking을 위해 truncation=False
        batched=True,
        num_proc=os.cpu_count() or 4,
        remove_columns=["text"],
    ).map(
        tokenize_and_chunk,
        batched=True,
    )
    print("데이터셋 전처리 완료.")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=config.output_model_path,
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        save_strategy="steps",
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=config.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    print("--- MLM 추가 학습 시작 ---")
    trainer.train()
    print("--- MLM 추가 학습 완료 ---")

    final_model_path = os.path.join(config.output_model_path, "final")
    trainer.save_model(final_model_path)
    print(f"최종 모델이 '{final_model_path}' 에 저장되었습니다.")


if __name__ == "__main__":
    # Config 객체 생성
    mlm_config = MLMConfig()

    # 생성된 Config를 사용하여 학습 함수 실행
    run_mlm_pretraining(mlm_config)

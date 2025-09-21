# preprocess.py (slang_dic 사전 제거, PyTorch 교정 모델 최종 버전)

import pandas as pd
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# pandas의 progress_apply를 사용하기 위한 설정
tqdm.pandas()

# 안정적인 최신 PyTorch 기반 맞춤법/띄어쓰기 교정 모델 로딩
# print("PyTorch 맞춤법/띄어쓰기 교정 모델을 로딩합니다... (시간이 소요될 수 있습니다)")
# MODEL_NAME = "paust/pko-t5-base"  # PyKoSpacing 대체...
# SPELLING_MODEL_LOADED = False
# try:
#     spelling_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     spelling_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     spelling_model.to(device)
#     print(f"모델 로딩 완료. Device: {device}")
#     SPELLING_MODEL_LOADED = True
# except Exception as e:
#     print(
#         f"경고: 맞춤법 교정 모델 '{MODEL_NAME}'을 로드할 수 없습니다. 맞춤법 교정을 건너뜁니다. 오류: {e}"
#     )


# def correct_spelling_and_spacing(text):
#     """PyTorch 기반 모델을 사용하여 맞춤법 및 띄어쓰기를 교정합니다."""
#     if not SPELLING_MODEL_LOADED or not isinstance(text, str) or not text:
#         return text

#     try:
#         input_text = "spell: " + text
#         input_ids = spelling_tokenizer.encode(
#             input_text, return_tensors="pt", max_length=256, truncation=True
#         ).to(device)

#         outputs = spelling_model.generate(
#             input_ids, max_length=256, num_beams=5, early_stopping=True
#         )

#         corrected_text = spelling_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return corrected_text
#     except Exception:
#         return text


def preprocess(text):
    """PyTorch 맞춤법/띄어쓰기 교정 및 정규식을 사용한 전처리 함수."""
    if not isinstance(text, str):
        return ""

    # # 1단계: PyTorch 모델을 이용한 맞춤법 및 띄어쓰기 교정
    # text = correct_spelling_and_spacing(text)

    # 2단계: 단독 사용 'ㅗ' 처리 및 불필요한 특수 문자 제거
    text = re.sub(r"(^|\s)[ㅗ]($|\s)", " 모욕 ", text)
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9.,!?&\s]", " ", text)

    # 3단계: 최종적으로 여러 개의 공백을 하나로 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    """CSV와 TXT 파일을 모두 전처리하고 기존 파일을 덮어씁니다."""
    dataset_dir = "../NIKL_AU_2023_COMPETITION_v1.0"
    filenames = ["train.csv", "dev.csv", "test.csv", "combined.txt"]

    for filename in filenames:
        file_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(file_path):
            print(f"경고: {file_path}를 찾을 수 없습니다. 건너뜁니다.")
            continue

        print(f"--- 전처리 시작 (Slang 사전 제거 버전): {file_path} ---")

        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension == ".csv":
            df = pd.read_csv(file_path)
            df["input"] = df["input"].progress_apply(preprocess)
            df.dropna(subset=["input"], inplace=True)
            df = df[df["input"].str.len() > 0]
            df.to_csv(file_path, index=False, encoding="utf-8")

        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            temp_df = pd.DataFrame(lines, columns=["input"])
            processed_lines = temp_df["input"].progress_apply(preprocess)
            processed_lines = processed_lines[processed_lines.str.len() > 0]

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(processed_lines))

        else:
            print(f"경고: 지원하지 않는 파일 형식입니다: {filename}. 건너뜁니다.")
            continue

        print(f"--- 전처리 완료 및 저장: {file_path} ---")


if __name__ == "__main__":
    main()

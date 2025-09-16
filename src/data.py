import os
import pandas as pd
import torch
import re
from konlpy.tag import Okt

# ================================================================== #
# 1. 전처리 함수 및 Okt 객체 정의
# Okt 객체는 로딩에 시간이 걸리므로 스크립트 상단에서 한 번만 생성.
# ================================================================== #
okt = Okt()

def preprocess_and_clean(text):
    """
    input 컬럼의 텍스트를 정제하는 함수.
    1. '&...&' 형태의 특수기호 및 기타 불필요한 문자 제거
    2. Okt 형태소 분석을 통한 불용어 제거
    """
    if not isinstance(text, str):
        return "" # 혹시 모를 비문자열 데이터 처리

    # 1단계: 정규 표현식을 이용한 특수기호 제거
    # &word&, &word.. 와 같은 패턴 제거
    text = re.sub(r'&\w+&', '', text)
    # 한글, 영어, 숫자, 공백을 제외한 모든 문자 제거
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]', '', text)
    
    # 2단계: Okt 형태소 분석 및 불용어(조사, 어미, 구두점) 제거
    clean_words = []
    for word, pos in okt.pos(text, stem=True): 
        if pos not in ['Josa', 'Eomi', 'Punctuation']:
            clean_words.append(word)
            
    return ' '.join(clean_words)

# ============================ 전처리 및 데이터 정리 추가완료 ====== #

class hate_dataset(torch.utils.data.Dataset):
    """dataframe을 torch dataset class로 변환"""

    def __init__(self, hate_dataset, labels):
        self.dataset = hate_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_data(dataset_dir):
    """csv file을 dataframe으로 load"""
    dataset = pd.read_csv(dataset_dir)
    print("dataframe 의 형태")
    print("-" * 100)
    print(dataset.head())
    return dataset


def construct_tokenized_dataset(dataset, tokenizer, max_length):
    """입력값(input)에 대하여 토크나이징"""
    print("tokenizer 에 들어가는 데이터 형태")
    print(dataset["input"][:5])

    tokenized_senetences = tokenizer(
        dataset["input"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        # return_token_type_ids=False,  # BERT 이후 모델(RoBERTa 등) 사용할때 False
    )
    print("tokenizing 된 데이터 형태")
    print("-" * 100)
    print(tokenized_senetences[:5])
    return tokenized_senetences


def prepare_dataset(dataset_dir, tokenizer, max_len):
    """학습(train)과 평가(test)를 위한 데이터셋을 준비"""
    # load_data
    train_dataset = load_data(os.path.join(dataset_dir, "train.csv")) 
    valid_dataset = load_data(os.path.join(dataset_dir, "dev.csv"))
    test_dataset = load_data(os.path.join(dataset_dir, "test.csv"))
    print("--- data loading Done ---")

    # split label
    train_label = train_dataset["output"].values
    valid_label = valid_dataset["output"].values
    test_label = test_dataset["output"].values

    # tokenizing dataset
    tokenized_train = construct_tokenized_dataset(train_dataset, tokenizer, max_len)
    tokenized_valid = construct_tokenized_dataset(valid_dataset, tokenizer, max_len)
    tokenized_test = construct_tokenized_dataset(test_dataset, tokenizer, max_len)
    print("--- data tokenizing Done ---")

    # make dataset for pytorch.
    hate_train_dataset = hate_dataset(tokenized_train, train_label)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_label)
    hate_test_dataset = hate_dataset(tokenized_test, test_label)
    print("--- pytorch dataset class Done ---")

    return hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset

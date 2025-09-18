import os
import pandas as pd
import torch


class hate_dataset(torch.utils.data.Dataset):
    """dataframe을 torch dataset class로 변환"""

    def __init__(self, hate_dataset, labels):
        self.dataset = hate_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        
        try:
            item["labels"] = torch.tensor(self.labels[idx])
        except Exception:
            pass
        
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
        dataset["input"].fillna("").tolist(),
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

import json
import csv
import pandas as pd

# 변환할 대상 파일들을 리스트로 정의
input_files = [
    '../NIKL_AU_2023_COMPETITION_v1.0/nikluge-au-2022-test.jsonl',
    '../NIKL_AU_2023_COMPETITION_v1.0/nikluge-au-2022-train.jsonl',
    '../NIKL_AU_2023_COMPETITION_v1.0/nikluge-au-2022-dev.jsonl'
]
# 출력 경로 지정. data.py 파일에서 정의된 파일명으로 변경.
output_files = [
    '../NIKL_AU_2023_COMPETITION_v1.0/test.csv',
    '../NIKL_AU_2023_COMPETITION_v1.0/train.csv',
    '../NIKL_AU_2023_COMPETITION_v1.0/dev.csv'
]

# 한 번에 3개 파일 처리하는 반복문
for input_file, output_file in zip(input_files, output_files):
    data = []

    # JSONL 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            data.append(record)

    # CSV 파일로 쓰기(저장)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'input', 'output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"'{input_file}' 파일이 '{output_file}' 파일로 변환되었습니다. 다음 작업을 진행하세요.")
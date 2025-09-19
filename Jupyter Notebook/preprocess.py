import pandas as pd
import os
import re
from konlpy.tag import Okt
from tqdm import tqdm

tqdm.pandas()

def preprocess_and_clean(text):
    """input 컬럼의 텍스트를 정제하는 함수."""
    if not isinstance(text, str):
        return ""
    
    # 소문자 통일
    text = text.lower()
    
    # URL, 이메일, 특수문자 제거
    text = re.sub(r'http\S+|www\S+', ' ', text)  # URL
    text = re.sub(r'\S+@\S+', ' ', text)         # 이메일
    text = re.sub(r'[^가-힣a-z0-9&\sㅋㅎ]', ' ', text)  # 한글, 영문, 숫자, &, ㅋㅎ만 유지
    
    # 반복 문자 처리 (ㅋㅋㅋㅋ → ㅋㅋ, ㅎㅎㅎㅎ → ㅎㅎ)
    text = re.sub(r'(ㅋ)\1{1,}', 'ㅋㅋ', text)
    text = re.sub(r'(ㅎ)\1{1,}', 'ㅎㅎ', text)
    
    # 불필요한 한글 자모 제거 (ㄱㄴㄷ 등 단일 낱자)
    text = re.sub(r'(?<![가-힣])[ㄱ-ㅎㅏ-ㅣ](?![가-힣])', '', text)
    
    # 다중 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    #Okt 사용 안함에 따라 return text 추가
    return text

    # Okt 객체를 함수 내에서 생성 - 일단 전부 주석 처리
    # okt = Okt()
    # clean_words = []
    # for word, pos in okt.pos(text, stem=True):
    #     if pos not in ['Josa', 'Eomi', 'Punctuation']:
    #         clean_words.append(word)
    # return ' '.join(clean_words)


def main():
    """데이터셋을 전처리하고 기존 CSV 파일을 덮어씁니다."""
    dataset_dir = '../NIKL_AU_2023_COMPETITION_v1.0'
    filenames = ['train.csv', 'dev.csv', 'test.csv']
    
    for filename in filenames:
        # 입력 파일과 출력 파일 경로를 동일하게 설정
        file_path = os.path.join(dataset_dir, filename)
        
        df = pd.read_csv(file_path)
        
        # 'input' 컬럼에 전처리 함수 적용
        df['input'] = df['input'].progress_apply(preprocess_and_clean)
 
        
        # 원본 데이터에 비어있는 행(NaN)이 있을 경우 제거
        df.dropna(subset=['input'], inplace=True)
        
        # 전처리 후 내용이 아예 없어진 행(빈 문자열) 제거
        df = df[df['input'].str.len() > 0]
      
        
        # 출력파일 저장. 기존 파일 덮어쓰기 주의!
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"변환 완료: {file_path}")

if __name__ == '__main__':
    main()
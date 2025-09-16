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
    
    # 정규식을 이용한 특수기호 제거
    text = re.sub(r'&\w+&', '', text)
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]', '', text)
    
    # Okt 객체를 함수 내에서 생성
    okt = Okt()
    clean_words = []
    for word, pos in okt.pos(text, stem=True):
        if pos not in ['Josa', 'Eomi', 'Punctuation']:
            clean_words.append(word)
    return ' '.join(clean_words)

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
        
        # 출력파일 저장. 기존 파일 덮어쓰기 주의!
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"변환 완료: {file_path}")

if __name__ == '__main__':
    main()
from transformers import AutoTokenizer
import os

# 절대 경로로 변경된 디렉토리 경로 설정
path = "/data1/hslim/PycharmProjects/alignment-handbook/cache/hub/models--amd--AMD-OLMo-1B/snapshots/422518a083f87a6811fed4ef28a5729be87d4e95"
resolved_path = os.path.abspath(path)  # 절대 경로로 변환 (확실히 처리)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(resolved_path)

text = "This    is    a     test ."

# 토크나이저로 토큰화 및 디코딩
tokens = tokenizer(text, return_tensors="pt").input_ids
decoded_text = tokenizer.decode(tokens[0], clean_up_tokenization_spaces=True)
print("Decoded with cleanup:", decoded_text)

decoded_text_no_cleanup = tokenizer.decode(tokens[0], clean_up_tokenization_spaces=False)
print("Decoded without cleanup:", decoded_text_no_cleanup)

# UNK 토큰 출력
print("EOS Token:", tokenizer.eos_token)

# 토큰화 테스트
tokens = tokenizer("gromflomite")
print("Tokenized Output:", tokens)


text = "This    is    a     test    .    With  multiple spaces."
tokens = tokenizer(text, return_tensors="pt").input_ids

# cleanup=True
decoded_with_cleanup = tokenizer.decode(tokens[0], clean_up_tokenization_spaces=True)
print("Decoded with cleanup:", decoded_with_cleanup)

# cleanup=False
decoded_without_cleanup = tokenizer.decode(tokens[0], clean_up_tokenization_spaces=False)
print("Decoded without cleanup:", decoded_without_cleanup)
from transformers import AutoTokenizer

# 모델 이름에 사용 중인 모델 경로 또는 이름 입력
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# chat_template 속성 확인
if hasattr(tokenizer, "chat_template"):
    print("Chat template is supported:", tokenizer.chat_template)
else:
    print("Chat template is NOT supported by this tokenizer.")

special_tokens = ['<|im_start|>', '<|im_end|>']

for token in special_tokens:
    if token in tokenizer.get_vocab():
        print(f"{token} is already in the tokenizer.")
    else:
        print(f"{token} is not in the tokenizer.")


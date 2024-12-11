from transformers import AutoTokenizer

# 기존 토크나이저 로드

path = "/data1/hslim/PycharmProjects/alignment-handbook/cache/hub/models--amd--AMD-OLMo-1B/snapshots/422518a083f87a6811fed4ef28a5729be87d4e95"
tokenizer = AutoTokenizer.from_pretrained(path)

# 변경된 설정 적용
tokenizer.add_special_tokens({
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>"
})

# 저장
tokenizer.save_pretrained(path)

import os
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

# 환경 설정
os.environ['HF_TOKEN'] = ""  # Hugging Face API 토큰 (필요한 경우 입력)

# FastAPI 객체 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로딩 (모델 이름을 EXAONE으로 설정)
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"  # EXAONE 모델로 설정
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # trust_remote_code=True 추가
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True  # trust_remote_code=True 추가
)

# 모델을 CPU로 설정
device = torch.device("cpu")
model.to(device)


# 응답 클리닝 함수
def clean_response(prompt: str, decoded_output: str) -> str:
    response = decoded_output.replace(prompt, "").strip()  # 프롬프트 제외
    if response.startswith("요."):
        response = response[2:].strip()  # 불필요한 부분 잘라내기
    return response


# 모델로 응답 생성
def generate_gemma_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # CPU로 데이터 이동
    with torch.no_grad():  # 그래디언트 계산 비활성화
        outputs = model.generate(
            **inputs,
            max_length=1024,  # 응답 길이 제한
            pad_token_id=tokenizer.eos_token_id,  # EOS 토큰 ID
            repetition_penalty=1.2,  # 반복 억제
            no_repeat_ngram_size=5,  # 
            temperature=0.7,  # 다양성
            top_p=0.9  # 상위 p 확률에서 샘플링
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)  # 디코딩
    final_response = clean_response(prompt, decoded)  # 응답 클리닝
    # 여기서부터는 "설명" 형식으로 답을 바꿔서 출력
    explanation = f"이 문제에 대한 설명: {final_response}. 질문의 답변을 기반으로 설명합니다."

    return explanation


# 요청 모델 정의
class PromptRequest(BaseModel):
    prompt: str


# 기본 GET 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


# POST 엔드포인트 (응답 생성)
@app.post("/chat")
async def chat(prompt_request: PromptRequest):
    prompt = prompt_request.prompt
    result = generate_gemma_response(prompt)  # 응답 생성
    return {"response": result}


# 서버 실행
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
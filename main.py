import os
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn


os.environ['HF_TOKEN'] = ""


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

device = torch.device("cpu")
model.to(device)


def clean_response(prompt: str, decoded_output: str) -> str:
    response = decoded_output.replace(prompt, "").strip()
    if response.startswith("요."):
        response = response[2:].strip()


def generate_gemma_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():  #
        outputs = model.generate(
            **inputs,
            max_length=1024,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,

        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_response = clean_response(prompt, decoded)
    return final_response


class PromptRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/chat")
async def chat(prompt_request: PromptRequest):
    prompt = prompt_request.prompt
    result = generate_gemma_response(prompt)  # 응답 생성
    return {"response": result}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


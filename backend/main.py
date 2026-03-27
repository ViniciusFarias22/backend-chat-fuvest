from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import httpx

# Carrega variáveis de ambiente
load_dotenv()

app = FastAPI(title="FUVEST Prep Chat API - Groq")

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    print("⚠️ AVISO: GROQ_API_KEY não encontrada no .env")

# Modelos disponíveis no Groq (gratuitos e bons)
AVAILABLE_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "llama-3.3-70b-versatile"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1500

class ChatResponse(BaseModel):
    response: str
    model: str

# Prompt do sistema (professor de FUVEST)
SYSTEM_PROMPT = """Você é um professor particular especializado em ajudar estudantes do ensino médio que estão se preparando para vestibulares como FUVEST, ENEM e outros.

Suas funções são:
1. EXPLICAR conteúdos escolares de forma clara e didática
2. CRIAR questões para os alunos praticarem
3. CORRIGIR respostas dos alunos
4. Dar DICAS de estudo

Matérias: Matemática, Português, Física, Química, Biologia, História, Geografia, Redação, etc.

Seja paciente, motivador e responda sempre em português do Brasil."""

@app.get("/")
async def root():
    return {"status": "online", "message": "FUVEST Prep Chat API - Groq", "model": "llama-3.3-70b"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "api_key_configured": bool(GROQ_API_KEY),
        "default_model": "llama-3.3-70b-versatile"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada. Verifique o arquivo .env")

        # Força um modelo válido
        if request.model not in AVAILABLE_MODELS:
            request.model = "llama-3.3-70b-versatile"

        # Prepara mensagens com system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        # Chama API do Groq
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": request.model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                },
                timeout=60.0
            )

            if response.status_code != 200:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("error", {}).get("message", response.text)
                raise HTTPException(status_code=response.status_code, detail=f"Erro Groq: {error_msg}")

            data = response.json()
            bot_response = data["choices"][0]["message"]["content"]

            return ChatResponse(response=bot_response, model=request.model)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Tempo limite excedido. Tente novamente.")
    except Exception as e:
        print(f"Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
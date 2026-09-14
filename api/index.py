from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import sys
import json
import joblib
from pathlib import Path
from google import genai

# Add src to sys.path so we can import agent logic
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import build_payload, generate_draft, apply_gate, SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

# Load retriever on startup
try:
    index_path = ROOT / "artifacts/retrieval/index.joblib"
    retriever = joblib.load(index_path)
except Exception as e:
    retriever = None
    print(f"Warning: Failed to load retriever: {e}")

api_key = os.getenv("GEMINI_API_KEY", "")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

class Turn(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: List[Turn] = []

@app.post("/api/chat")
def chat(request: ChatRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not loaded")

    # Format into the expected structure
    example = {
        "customer_message": request.message,
        "prior_context": [{"role": t.role, "text": t.text} for t in request.history]
    }
    
    # Get evidence
    evidence = retriever.search(example["customer_message"], k=3)
    payload = build_payload(example, evidence)

    client = genai.Client(api_key=api_key)
    
    try:
        draft, raw_text, usage = generate_draft(
            client,
            model_name,
            json.dumps(payload, ensure_ascii=False)
        )
        
        result = apply_gate(draft, evidence)
        
        return {
            "intent": result.get("intent", "other_unclear"),
            "reply": result.get("reply", ""),
            "decision": result.get("decision", "ESCALATE"),
            "risk_flags": result.get("risk_flags", []),
            "evidence_used": len(evidence) > 0,
            "gate_reasons": result.get("gate_reason_codes", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {"status": "ok", "retriever_loaded": retriever is not None}

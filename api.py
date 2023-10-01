from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request

app = FastAPI()

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

@app.post("/embed")
async def get_embeddings(request: Request):
    data = await request.json()
    text = data.get("text")
    embeddings = model.encode([text], normalize_embeddings=True)
    return {"embeddings": embeddings.tolist()}

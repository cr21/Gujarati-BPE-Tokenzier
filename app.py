from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from encoder import BPEGujaratiTokenizer
from fastapi.middleware.cors import CORSMiddleware

# Define a Pydantic model for the request body
class EncodeRequest(BaseModel):
    text: str

class DecodeRequest(BaseModel):
    tokens: str

# Initialize the tokenizer
tokenizer = BPEGujaratiTokenizer(corpus_path="gu_corpus.txt", max_vocab_size=5000, sample_size=20000)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/encode")
async def encode_text(request: EncodeRequest):
    """Encodes the input text and returns the tokens."""
    print("request.text: ", request.text)
    return {"encoded_tokens": tokenizer.encode(request.text)}

@app.post("/decode")
async def decode_tokens(request: DecodeRequest):
    """Decodes the input tokens and returns the original text."""
    print(request.tokens)
    tokens = request.tokens.split(',')
    tokens = list(map(int, tokens))
    decoded_text = tokenizer.decode(tokens)
    return {"decoded_text": decoded_text}

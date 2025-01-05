import gradio as gr
from encoder import BPEGujaratiTokenizer

# Initialize the tokenizer
tokenizer = BPEGujaratiTokenizer(corpus_path="gu_corpus.txt", max_vocab_size=5000, sample_size=20000)

def encode_text(text):
    """Encodes the input text and returns the tokens."""
    return tokenizer.encode(text)

def decode_tokens(tokens):
    """Decodes the input tokens and returns the original text."""
    return tokenizer.decode(tokens.split(','))

# Create the Gradio interface for encoding
encode_iface = gr.Interface(
    fn=encode_text,  # Directly use the encode_text function
    inputs=gr.Textbox(label="Input Text", placeholder="Enter text to encode..."),
    outputs=gr.Textbox(label="Encoded Tokens"),
    title="Basic Gujarati BPE Tokenizer (5000 Vocab)",
    description="This app encodes Gujarati text into tokens using BPE."
)

# Create the Gradio interface for decoding
decode_iface = gr.Interface(
    fn=decode_tokens,  # Directly use the decode_tokens function
    inputs=gr.Textbox(label="Input Tokens (comma-separated)", placeholder="Enter tokens to decode..."),
    outputs=gr.Textbox(label="Decoded Text"),
    title="Basic Gujarati BPE Tokenizer (5000 Vocab)",
    description="This app decodes tokens back into text using BPE."
)

# Launch both interfaces
if __name__ == "__main__":
    encode_iface.launch()
    decode_iface.launch()

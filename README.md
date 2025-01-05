---
title: Gujarati BPE Tokenizer
emoji: 🇮🇳
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Gujarati BPE Tokenizer

A FastAPI-based web application that provides Byte Pair Encoding (BPE) tokenization for Gujarati text.

## Features
- Encode Gujarati text to BPE tokens
- Decode BPE tokens back to Gujarati text
- Web interface for easy interaction



---

## What is Byte Pair Encoding (BPE)?

Byte Pair Encoding (BPE) is compression-based technique to create a fixed-size vocabulary by breaking down text into subword units. It is widely used in Natural Language Processing (NLP) to handle rare words and improve efficiency in tokenizing languages. It's language agnostic and can be used for any language.

### How Does BPE Work?

BPE starts with individual characters as tokens and iteratively merges the most frequent pairs of tokens until the desired vocabulary size is reached. This allows the algorithm to handle common words as a single token and break rare or unknown words into smaller, more manageable pieces.

### Example

Suppose we have the following simple corpus:

```
aaabdaaabac
```

**Step 1**: Start with individual characters as tokens:  
```
['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']
```
**Step 2.a**: Create  adjacent token pairs:  
```
[('a', 'a'), ('a', 'a'), ('a', 'b'), ('b', 'd'), ('d', 'a'), ('a', 'a'), ('a', 'b'), ('b', 'a'), ('a', 'c')]
```
**Step 2.b**: Count the frequency of adjacent token pairs:  
- `('a', 'a')`: 4 occurrences  
- `('a', 'b')`: 2 occurrences  
- `('b', 'd')`: 1 occurrence  
- `('d', 'a')`: 1 occurrence  
- `('a', 'c')`: 1 occurrence  

**Step 3**: Merge the most frequent pair (`'a', 'a'`) into a single token:  
```
['aa', 'a', 'b', 'd', 'aa', 'a', 'b', 'a', 'c']
```

**Step 4**: Repeat the process (2.2a to 3):  
- New frequencies:  
  - `('aa', 'a')`: 2  
  - `('a', 'b')`: 2  
  - `('b', 'd')`: 1  
  - `('d', 'aa')`: 1  
  - `('a', 'c')`: 1  

- Merge `'aa', 'a'` into `'aaa'`:  
```
['aaa', 'b', 'd', 'aaa', 'b', 'a', 'c']
```

**Step 5**: Continue until the vocabulary reaches the desired size.

### Final Vocabulary

After a few iterations, we might get a vocabulary like this:  
```
['a', 'b', 'c', 'd', 'aa', 'aaa']
```

### Why Use BPE?

- **Handles Rare Words**: BPE breaks rare or unknown words into smaller, meaningful subword tokens.  
- **Compact Vocabulary**: A smaller vocabulary is easier to manage and requires less memory.  
- **Improved Generalization**: Subwords help in understanding unknown or morphologically complex words.

This approach is particularly useful for languages like Gujarati, where words can have rich inflections and morphology.

--- 


Byte Pair Encoding (BPE) handles unknown words effectively by breaking them into smaller, known subword units, ensuring that even unseen words can be represented using the existing vocabulary. Here's a simple explanation of how it works:

### Handling Unknown Words in BPE

When a word is not in the tokenizer's vocabulary (i.e., it's an **unknown word**), the BPE algorithm breaks it down into smaller subword tokens that **are** in the vocabulary. This ensures that no word is truly "unknown."

### Example

#### Vocabulary
Let's say the tokenizer has been trained with the following vocabulary:
```
['a', 'b', 'c', 'ab', 'abc']
```

#### Input Word
We encounter a new word:  
```
'abcd'
```

#### Tokenization Process
1. **Start with Characters**:  
   Break the word into individual characters:  
   ```
   ['a', 'b', 'c', 'd']
   ```

2. **Merge Known Subwords**:  
   - Look for the longest subword in the vocabulary that matches the input sequence.
   - `'ab'` is in the vocabulary, so merge `'a'` and `'b'`:  
     ```
     ['ab', 'c', 'd']
     ```

3. **Continue Matching**:  
   - `'abc'` is in the vocabulary, so merge `'ab'` and `'c'`:  
     ```
     ['abc', 'd']
     ```

4. **Handle Remaining Characters**:  
   - `'d'` is not in the vocabulary, so it remains as a single token.  

Final Tokenization:  
```
['abc', 'd']
```

### Key Points
1. **Fallback to Subwords**: Unknown words are split into smaller subwords or individual characters until they can be represented using the vocabulary.    
2. **Efficient Representation**: By leveraging subwords, BPE can represent new, complex, or rare words without requiring a massive vocabulary.  

### Why is This Useful?

For languages like Gujarati, which have complex morphology, BPE helps handle unseen words effectively by leveraging subwords, making it robust for real-world applications like machine translation or text classification.


## Basic Implementation of BPE

```python

def read_corpus(corpus_path:str):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text



class BPEGujaratiTokenizer:
    def __init__(self, corpus_path:str, max_vocab_size:int=5000, sample_size:int=20000):
        self.corpus = read_corpus(corpus_path)
        self.max_vocab_size = max_vocab_size
        self.corpus_vocab = sorted(list(set(self.corpus)))
        self.corpus_vocab_size = len(self.corpus_vocab)
        self.stoi = { ch:i for i,ch in enumerate(self.corpus_vocab) }
        self.itos = { i:ch for i,ch in enumerate(self.corpus_vocab) }
        self.sample_size = sample_size

        self.vocab, self.merges = self.train_bpe(self.corpus, self.max_vocab_size, self.sample_size)


    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts


    def merge(self,ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids



    def train_bpe(self, corpus, max_vocab_size, sample_size=None):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        if sample_size :
            corpus = corpus[:sample_size]
        num_merges = max_vocab_size - len(self.vocab)
        tokens = corpus.encode('utf-8')
        tokens= list(map(int, tokens))
        ids = list(tokens)
        self.merges = {} # (int, int) -> int
        print(f"Before training: ids length: {len(ids)}")
        print(f"Before training: tokens length: {len(tokens)}")
        print("Before training: merges length: ", len(self.merges))

        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = len(self.vocab)+i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
        # merge the vocab
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        print(f"After training: ids length: {len(ids)}")
        print(f"After training: tokens length: {len(tokens)}")
        print("After training: merges length: ", len(self.merges))
        print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
        return self.vocab, self.merges

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    
    def decode(self, tokens):
        tokens = b"".join(self.vocab[idx] for idx in tokens)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
import time
if __name__ == "__main__":
    start_time = time.time()
    tokenizer = BPEGujaratiTokenizer(corpus_path="gu_corpus.txt", max_vocab_size=5000, sample_size=20000)
    end_time = time.time()
    print(f"Time taken to train: {end_time - start_time} seconds")
    print("--------------------------------")
    start_time = time.time()
    print(tokenizer.encode("હું તને પ્રેમ કરું છું"))
    end_time = time.time()
    print(f"Time taken to encode: {end_time - start_time} seconds")
    print("--------------------------------")
    start_time = time.time()
    print(tokenizer.decode(tokenizer.encode("હું તને પ્રેમ કરું છું")))
    end_time = time.time()
    print(f"Time taken to decode: {end_time - start_time} seconds")
    print("--------------------------------")
    start_time = time.time()
    sentences = ["હું આજે ખૂબ ખુશ છું.","તું શું કરે છે? ","મને ચા પીવી છે. ","એ બધું સરસ છે. ","આ પુસ્તક ખૂબ રસપ્રદ છે. ","તારે ક્યારે આવવું છે? ","આ મારો મિત્ર છે. ","હું શાકભાજી લઈ આવ્યો છું. ","આકાશ માં વાદળ છે. ","શાળા ક્યારે શરૂ થશે? ",'આ પુસ્તક ખૂબ રસપ્રદ છે.']
    for sentence in sentences:
        print("original: ", sentence)
        print("encoded: ", tokenizer.encode(sentence))
        print("decoded: ", tokenizer.decode(tokenizer.encode(sentence)))
        print(tokenizer.decode(tokenizer.encode(sentence)) == sentence)
    end_time = time.time()
    print(f"Time taken to decode: {end_time - start_time} seconds")
    print("--------------------------------")
```

```
[292, 310, 164, 290, 363, 329, 325, 310, 155, 600]
Time taken to encode: 0.0006651878356933594 seconds
--------------------------------
હું તને પ્રેમ કરું છું
Time taken to decode: 0.0004611015319824219 seconds
--------------------------------
original:  હું આજે ખૂબ ખુશ છું.
encoded:  [292, 310, 1987, 150, 314, 172, 1804, 503, 600, 46]
decoded:  હું આજે ખૂબ ખુશ છું.
True
original:  તું શું કરે છે? 
encoded:  [279, 1700, 310, 412, 267, 155, 260, 63, 32]
decoded:  તું શું કરે છે? 
True
original:  મને ચા પીવી છે. 
encoded:  [274, 290, 154, 553, 549, 269, 155, 260, 46, 32]
decoded:  મને ચા પીવી છે. 
True
original:  એ બધું સરસ છે. 
encoded:  [479, 334, 343, 310, 184, 1538, 503, 260, 46, 32]
decoded:  એ બધું સરસ છે. 
True
original:  આ પુસ્તક ખૂબ રસપ્રદ છે. 
encoded:  [256, 134, 298, 280, 437, 294, 1990, 172, 316, 326, 1308, 361, 503, 260, 46, 32]
decoded:  આ પુસ્તક ખૂબ રસપ્રદ છે. 
True
original:  તારે ક્યારે આવવું છે? 
encoded:  [279, 344, 149, 482, 347, 1691, 155, 260, 63, 32]
decoded:  તારે ક્યારે આવવું છે? 
True
original:  આ મારો મિત્ર છે. 
encoded:  [256, 134, 1803, 283, 174, 366, 288, 503, 260, 46, 32]
decoded:  આ મારો મિત્ર છે. 
True
original:  હું શાકભાજી લઈ આવ્યો છું. 
encoded:  [292, 1700, 621, 418, 429, 1527, 388, 788, 413, 155, 600, 46, 32]
decoded:  હું શાકભાજી લઈ આવ્યો છું. 
True
original:  આકાશ માં વાદળ છે. 
encoded:  [256, 134, 294, 1089, 307, 285, 181, 405, 345, 503, 260, 46, 32]
decoded:  આકાશ માં વાદળ છે. 
True
original:  શાળા ક્યારે શરૂ થશે? 
encoded:  [330, 888, 391, 482, 182, 268, 1248, 165, 330, 260, 63, 32]
decoded:  શાળા ક્યારે શરૂ થશે? 
True
original:  આ પુસ્તક ખૂબ રસપ્રદ છે.
encoded:  [256, 134, 298, 280, 437, 294, 1990, 172, 316, 326, 1308, 361, 503, 260, 46]
decoded:  આ પુસ્તક ખૂબ રસપ્રદ છે.
True
Time taken to decode: 0.009427070617675781 seconds
--------------------------------
```

---

## Hugging Face Space

This project is hosted on [Hugging Face Spaces](https://huggingface.co/spaces), providing an interactive web interface to experiment with the Gujarati BPE Tokenizer. You can easily encode Gujarati text into tokens and decode tokens back into text.

### Features
- **Encode Gujarati Text**: Input any Gujarati text and get the corresponding BPE token IDs.  
- **Decode Token IDs**: Input a sequence of BPE token IDs and retrieve the original Gujarati text.  
- **Interactive Interface**: A user-friendly interface to try the tokenizer without setting up the environment locally.  

### Try it Here:
[Click to Open the Gujarati BPE Tokenizer Space](https://huggingface.co/spaces/crpatel/Gujarati-BPE-Tokenizer)  


### How to Use:
1. Open the Hugging Face Space link.
2. Use the provided input fields:
   - Enter Gujarati text to **Encode** it into token IDs.
   - Enter token IDs (comma-separated) to **Decode** them back into Gujarati text.
3. View the results directly on the interface.

### Example Use Case:
- **Input Text**: "દરેક સૂર્યાસ્ત એક પાઠ શીખવે છે, અને દરેક સૂર્યોદય આશાના કિરણ સાથે આવે છે!"  
- **Encoded Tokens**: `[361,268,356,312,411,268,286,1695,477,298,261,160,370,1259,282,267,155,260,44,397,166,268,356,312,411,268,286,1042,317,328,330,293,391,937,291,858,347,267,155,260,33]`  
- **Decoded Text**: "દરેક સૂર્યાસ્ત એક પાઠ શીખવે છે, અને દરેક સૂર્યોદય આશાના કિરણ સાથે આવે છે!"

![Hugging Face Space Interface](/static/hugging_space.png)

--- 
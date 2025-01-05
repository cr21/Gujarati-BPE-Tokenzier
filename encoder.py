
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
        print(f"before training vocab {len(self.vocab)}")
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
        print(f"After training vocab {len(self.vocab)}")
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
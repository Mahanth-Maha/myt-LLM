from tqdm import tqdm


def get_bigram_byte_frequency(ids):
    counts = {}
    for bigram in zip(ids[:-1], ids[1:]):
        counts[bigram] = counts.get(bigram, 0) + 1
    return counts

def get_max_frequency_bigram(ids):
    counts = {}
    max_pair = (0,(-1,-1))
    for bigram in zip(ids[:-1], ids[1:]):
        # bigram = str(bigram[0]) + ';' + str(bigram[1])
        counts[bigram] = counts.get(bigram,0) + 1
        if max_pair[0] < counts[bigram]:
            max_pair = (counts[bigram] , bigram)
    return max_pair[1]

def merge(ids, src, dest):
    res = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == src[0] and ids[i+1] == src[1]:
            res.append(dest)
            i += 1
        else:
            res.append(ids[i])
        i += 1
    return res


class myTokenizer:
    def __init__(self,special_tokens = None):
        self.merges = {}
        self.special_tokens = special_tokens  if special_tokens is not None else {}
        self.vocab = self._vocab()


    def train(self, text, vocab_size):
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))
        merges = {} 
        vocab = {idx: bytes([idx]) for idx in range(256)} 
        for i in tqdm(range(num_merges), desc='Training Progress',unit='merge'):
            source = get_max_frequency_bigram(ids)
            dest = 256 + i
            ids = merge(ids, source, dest)

            merges[source] = dest
            vocab[dest] = vocab[source[0]] + vocab[source[1]]

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair = get_max_frequency_bigram(ids)
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def _vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")    
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        
    def load(self, model_file):
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._vocab()
        
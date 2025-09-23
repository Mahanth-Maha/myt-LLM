from collections import Counter
from heapq import heapify, heappop, heappush
from tqdm import tqdm


def get_bigram_byte_frequency(ids):
    return Counter(zip(ids[:-1], ids[1:]))

class myTokenizer_faster:
    def __init__(self, special_tokens=None):
        self.merges = {}
        self.special_tokens = special_tokens if special_tokens else {}
        self.vocab = self._vocab()

    def train(self, text, vocab_size):
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))
        vocab = {idx: bytes([idx]) for idx in range(256)} 

        counts = get_bigram_byte_frequency(ids)
        heap = [(-freq, source) for source, freq in counts.items()]
        heapify(heap)

        for i in tqdm(range(num_merges), desc="Training", unit="merge"):
            while heap:
                freq, source = heappop(heap)
                if counts[source] == -freq and counts[source] > 0:
                    break
            else:
                break

            dest = 256 + i
            self.merges[source] = dest
            vocab[dest] = vocab[source[0]] + vocab[source[1]]

            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == source[0] and ids[j + 1] == source[1]:
                    new_ids.append(dest)

                    if len(new_ids) >= 2:
                        left = (new_ids[-2], dest)
                        counts[left] += 1
                        heappush(heap, (-counts[left], left))

                    if j + 2 < len(ids):
                        right = (dest, ids[j + 2])
                        counts[right] += 1
                        heappush(heap, (-counts[right], right))

                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
            counts[source] = 0

        self.vocab = vocab

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
    
    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while True:
            candidates = [(self.merges[bg], bg) for bg in zip(ids[:-1], ids[1:]) if bg in self.merges]
            if not candidates:
                break
            _, pair = min(candidates)
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and (ids[j], ids[j + 1]) == pair:
                    new_ids.append(self.merges[pair])
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
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
        
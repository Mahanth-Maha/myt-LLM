# Tokenization

> Byte Pair Encoding (BPE) Implementations

This folder contains module `bpe`, which has three versions of **Byte Pair Encoding (BPE)**, a subword tokenization algorithm widely used in NLP.

## Implementations

### 1. Plain BPE

* Repeatedly scans corpus to find and merge most frequent pair.
* **Complexity**: `O(T · (N + V²))`
* Simple but slow

### 2. BPE with Heap

* Maintains most frequent pair in a heap.
* **Complexity**: `O(N + T log V)`
* Much faster than plain BPE

### 3. Single-Pass Heap + Merge

* Optimized version that merges and updates in one pass.
* **Complexity**: `O(N + T log V)`
* Fastest but most complex

## Comparison
see [ [01_Dataset.ipynb](./dev/01_tokenizatio/01_Dataset.ipynb) ] Notebook for comparison on wikitext dataset

|bpe-type| Avg merge/s | Time taken | Time complexity|
|---|---|---|---|
| bpe_plain | 1.00 merge/s | 10 min  | $O(n * \text{merges})$ |
| bpe_fast | 1.15 merge/s | 4 min   | $O(\text{merges} \log(\text{merges}))$ |
| bpe_faster | 1.65 merge/s | 2 min  | $O(\text{merges} \log(\text{merges}))$ |


## Tokenizers in Practice

| Tokenizer                     | Vocab Size | Models Using It           |
| ----------------------------- | ---------- | ------------------------- |
| **GPT-2 BPE**                 | 50,257     | GPT-2, early GPT-3        |
| **GPT-3.5/4 (`cl100k_base`)** | 100,257    | GPT-3.5, GPT-4 (ChatGPT)  |
| **LLaMA SentencePiece**       | 32,000     | LLaMA-1, LLaMA-2, LLaMA-3 |
| **T5 SentencePiece**          | 32,128     | T5, mT5                   |
| **BERT WordPiece**            | 30,522     | BERT, RoBERTa variants    |


## References

* [Sennrich et al., 2015](https://arxiv.org/abs/1508.07909) – BPE for NMT
* [OpenAI tiktoken](https://github.com/openai/tiktoken)
* [Google SentencePiece](https://github.com/google/sentencepiece)


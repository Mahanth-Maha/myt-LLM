# Byte pair Encoding


### Comparison btw versions : 

refer [ [dev/01_tokenization/01_Dataset.ipynb](./dev/01_tokenization/01_Dataset.ipynb) ]  for details


|bpe-type| Avg merge/s | Time taken | Time complexity|
|---|---|---|---|
| bpe_plain | 1.00 merge/s | 10 min  | $O(n * \text{merges})$ |
| bpe_fast | 1.15 merge/s | 4 min   | $O(\text{merges} \log(\text{merges}))$ |
| bpe_faster | 1.65 merge/s | 2 min  | $O(\text{merges} \log(\text{merges}))$ |


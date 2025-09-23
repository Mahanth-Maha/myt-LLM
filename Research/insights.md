

## useful sites or links 

* Visualize GPT: [bbycroft.net/llm](https://bbycroft.net/llm)

# Design 

## 01 Tokenization

1. tiktoken is best starting point but its not in 2 powers :(
1. sentencepiece can be used to train my own but takes more time :(

* `Solution:` Extend existing method  :)

## 02 Pre-Training 

1. Insights/Problems 
 
* `Solution:` 


# Random: understanding few topics step by step

## Working Steps MultiHeadSelfAttention in GPT (example shapes flow)

* `context_length = 8`
* batch size `B = 2`
* embedding dimension `d_model = 4`
* number of heads `h = 2` $\rightarrow$ so each head has `d_head = 2`
* sequence length fed in: `T = 6` (so `T ≤ context_length`)


### Input tokens

```python
X: (B=2, T=6)   # token IDs
```

Example:

```python
X =
[[10, 25, 502, 1, 77, 3],
 [5, 18, 20, 111, 42, 7]]
```


### Token embeddings

```python
token_embeddings: (B=2, T=6, d_model=4)
```


### Positional embeddings

```python
self.position_embeddings = nn.Embedding(context_length=8, d_model=4)
```

let's take positions `[0,1,2,3,4,5]` since `T=6`:

```python
pos_embeddings: (T=6, d_model=4)
```

Broadcasted to batch:

```python
pos_embeddings: (B=2, T=6, d_model=4)
```


### Add token + position embeddings

```python
embeds = token_embeddings + pos_embeddings
shape: (B=2, T=6, d_model=4)
```


### QKV projection (combined linear)

From:

```python
self.qkv = nn.Linear(d_model, 3*d_model)
```

Input:

```python
embeds: (B=2, T=6, d_model=4)
```

Output:

```python
qkv: (B=2, T=6, 3*d_model) = (2, 6, 12)
```


### Split Q, K, V

```python
Q: (2, 6, 4)
K: (2, 6, 4)
V: (2, 6, 4)
```


### Reshape into heads

Since `h=2, d_head=2`:

```python
Q: (B=2, h=2, T=6, d_head=2)
K: (2, 2, 6, 2)
V: (2, 2, 6, 2)
```


### Scaled Dot-Product Attention

Scores:

$$
\text{Scores} = \frac{Q K^T}{\sqrt{d_{head}}}
$$

```python
Scores: (B=2, h=2, T=6, T=6) = (2, 2, 6, 6)
```

If **causal mask** is applied $\rightarrow$ upper triangle masked.

Softmax:

```python
A: (2, 2, 6, 6)
```


### Weighted sum

```python
Context = A @ V
shape: (2, 2, 6, 2)
```


### Concatenate heads

Reshape back:

```python
Context: (B=2, T=6, d_model=4)
```


### Final projection

```python
Out = Context @ W_O
Out: (2, 6, 4)
```

### Final Output of MHA block (for `T=6`) is:

```python
Out: (B=2, T=6, d_model=4)
```

But all the internal machinery was built to allow **any T ≤ context\_length**.


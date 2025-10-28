# Development Arena of myT-LLM

## Road Map

### Implemented

* utils
  - [x] Tokenization [ [dev/01_tokenization](./dev/01_tokenization) ]
  - [x] Fast Tokenization - Encode Large Data
  - [x] Trainer 
  - [x] Logger
  - [x] Checks for Pre-training 
    - [x] Loss Check
    - [x] metrics : perplexity, loss, acc, bpc
    - [x] Generation Check
  - [x] Architectural (CORE) - Embed tying (-> Needs Residual Norm)
  - [x] Architectural (CORE) - Residual Normalization
  - [x] Architectural (CORE) - Multi Query Attn Optimization / Vectorization
  - [x] Architectural (CORE) - Single. vs. Multi. vs Grouped. Query Attn in MHA
  - [x] Architectural - hidden-dimension

- [x] Transformer (continuos updates on this)
  - [x] Architectural - non-linearity
  - [x] Architectural (CORE) - Norms : Layer vs RMS


* utils 
  - [x] DataLoader 
  - [x] Visualization
  - [x] save checkpoints (last K and few milestones)

* Optimizations 
  - [x] Architectural - Tokenizer fix (such as merging two words, not able to merge spaces, int as int - don't merge, etc)
  - [x] Architectural - RoPE
  - [x] Training - optimizer and settings
  - [x] Training - clip grads
  - [x] Training - lr : decay and warmups 
  - [x] Systems - Choose Precision 
  - [x] Systems - Compile 

* Improvements
  - [x] Architectural (CORE) - Attn (Flash)

### In Progress
- None

### To Implement
- Done !


(will implement new features soon!)
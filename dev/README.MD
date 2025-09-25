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

### In Progress

- [x] Transformer (continuos updates on this)
  - [ ] Architectural - non-linearity
  - [ ] Architectural (CORE) - Norms : Layer vs RMS

### To Implement

* utils 
  - [ ] DataLoader 
  - [ ] Visualization
  - [ ] save checkpoints (last K and few milestones)

* Optimizations 
  - [ ] Architectural - Tokenizer fix (such as merging two words, not able to merge spaces, int as int - don't merge, etc)
  - [ ] Architectural - RoPE
  - [ ] Training - optimizer and settings
  - [ ] Training - clip grads
  - [ ] Training - lr : decay and warmups 
  - [ ] Systems - Choose Precision 
  - [ ] Systems - Compile 

* Improvements

  - [ ] Architectural (CORE) - Attn (Flash)

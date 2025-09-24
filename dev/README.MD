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

### In Progress

- [x] Transformer (continuos updates on this)

### To Implement

* utils 
  - [ ] DataLoader 
  - [ ] Visualization
  - [ ] save checkpoints (last K and few milestones)

* Optimizations 

  - [ ] Architectural - Tokenizer fix (such as merging two words, not able to merge spaces, int as int - don't merge, etc)
  - [ ] Architectural - Embbed tying
  - [ ] Architectural - Residual Norm
  - [ ] Training - optimizer and settings
  - [ ] Training - clip grads
  - [ ] Training - lr : decay and warmups 
  - [ ] Systems - Choose Precision 
  - [ ] Systems - Compile 

* Improvements

  - [ ] Architectural (CORE) - GQA in MHSA
  - [ ] Architectural (CORE) - Attn (Flash)
  - [ ] Architectural (CORE) - RoPE
  - [ ] Architectural (CORE) - Norms : Layer vs RMS
  - [ ] Architectural (CORE) - non-linearity

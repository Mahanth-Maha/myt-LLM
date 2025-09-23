# Development Arena of myT-LLM

## Road Map

### Implemented

- [x] Tokenization [ [dev/01_tokenization](./dev/01_tokenization) ]

### In Progress

- [x] Transformer 

### To Implement

* utils 

  - [ ] DataLoader 
  - [ ] Trainer 
  - [ ] Logger
  - [ ] Visualization
  - [ ] Checks for Pre-training 
    - [ ] Loss Check
    - [ ] metrics : perplexity, etc
    - [ ] Generation Check

* Optimizations 

  - [ ] Architectural - tokenizer fix
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

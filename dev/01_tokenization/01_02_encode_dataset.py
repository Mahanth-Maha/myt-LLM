# %% [markdown]
# # Encode Train, Val, Test splits

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# %%
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.load("wikitext_spm.model")

print(f'[+] Tokenizer loaded')
# %%
from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
ds

# %%
def get_text(typ = ds['train']):
    all_text = ''
    for row in typ:
        if row['text']:
            all_text += row['text']
    return all_text 

def get_encoded_text(typ = ds['train']):
    encs = []
    for row in typ:
        if row['text']:
            encs.extend(tokenizer.encode(row['text']))
    return encs


# %%
print(f'[+] Train :')
# Xtr = torch.tensor(tokenizer.encode(get_text(ds['train'])), dtype=torch.long)
# Xtr.shape
Xtr = torch.tensor(get_encoded_text(ds['train']), dtype=torch.long)
Xtr.shape

# %%
# Xval = torch.tensor(tokenizer.encode(get_text(ds['validation'])), dtype=torch.long)
# Xval.shape
print(f'[+] Val :')
Xval = torch.tensor(get_encoded_text(ds['validation']), dtype=torch.long)
Xval.shape

# %%
# Xte = torch.tensor(tokenizer.encode(get_text(ds['test'])), dtype=torch.long)
# Xte.shape
print(f'[+] Test :')
Xte = torch.tensor(get_encoded_text(ds['test']), dtype=torch.long)
Xte.shape

# %%
Xtr.shape, Xval.shape, Xte.shape

# %%
import os 
os.makedirs('saved_tensors',exist_ok=True)

torch.save(Xtr, 'saved_tensors/Xtr.pt')
torch.save(Xval, 'saved_tensors/Xval.pt')
torch.save(Xte, 'saved_tensors/Xte.pt')

print(f'[+] Saved! Done')

import tiktoken
import torch
import torch.nn.functional as F
import random

# tokens

VALID_LAST_TOKEN = 100255

# ck Base defined

ENDOFTEXT = "<|endoftext|>"
ENDOFTEXT_IDX = 100257
FIM_PREFIX = "<|fim_prefix|>"
FIM_PREFIX_IDX = 100258
FIM_MIDDLE = "<|fim_middle|>"
FIM_MIDDLE_IDX = 100259
FIM_SUFFIX = "<|fim_suffix|>"
FIM_SUFFIX_IDX = 100260

ENDOFPROMPT = "<|endofprompt|>"
ENDOFPROMPT_IDX = 100276

# MY TOKENS

MAHAFILESEP = "<|maha_sep|>"
MAHAFILESEP_IDX = 100264

UNK_TOKEN = "<|unknown|>"
UNK_TOKEN_IDX = 100351          # dont use for anything else

spl_tok_set = {
    MAHAFILESEP,
    UNK_TOKEN,
}

spl_tok_dict = {
    MAHAFILESEP: MAHAFILESEP_IDX,
    UNK_TOKEN: UNK_TOKEN_IDX,
}

all_spl_toks_set = {
    ENDOFTEXT,
    FIM_PREFIX,
    FIM_MIDDLE,
    FIM_SUFFIX,
    ENDOFPROMPT,
    MAHAFILESEP,
    UNK_TOKEN,
}

all_spl_toks_idx_set = {
    ENDOFTEXT_IDX,
    FIM_PREFIX_IDX,
    FIM_MIDDLE_IDX,
    FIM_SUFFIX_IDX,
    ENDOFPROMPT_IDX,
    MAHAFILESEP_IDX,
    UNK_TOKEN_IDX,
}

all_spl_toks_dict = {
    ENDOFTEXT: ENDOFTEXT_IDX,
    FIM_PREFIX: FIM_PREFIX_IDX,
    FIM_MIDDLE: FIM_MIDDLE_IDX,
    FIM_SUFFIX: FIM_SUFFIX_IDX,
    ENDOFPROMPT: ENDOFPROMPT_IDX,
    MAHAFILESEP: MAHAFILESEP_IDX,
    UNK_TOKEN: UNK_TOKEN_IDX,
}



def get_encoder():
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.Encoding(
        name="cl100k_maha",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            MAHAFILESEP: MAHAFILESEP_IDX,
            UNK_TOKEN: UNK_TOKEN_IDX,
        }
    )
    return enc


class mytlmTokenizer:
    def __init__(self, special_tokens, vocab_size = None, base_encoding = "cl100k_base", my_enc_name="cl100k_maha"):
        self.special_tokens = special_tokens
        self.base_enc = tiktoken.get_encoding(base_encoding)

        self.encoder = tiktoken.Encoding(
            name=my_enc_name,
            pat_str=self.base_enc._pat_str,
            mergeable_ranks=self.base_enc._mergeable_ranks,
            special_tokens={
                **self.base_enc._special_tokens, 
                **self.special_tokens
            },
        )
        
        self.n_vocab = vocab_size if vocab_size is not None else self.encoder.n_vocab
        
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}

    def encode(self, text):
        return self.encoder.encode(
            text,
            allowed_special="all",
            disallowed_special=()
        )

    def decode(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        tokens = [t if (t <= VALID_LAST_TOKEN or t in self.special_tokens.values()) else UNK_TOKEN_IDX for t in tokens]
        return self.encoder.decode(tokens)

    @torch.no_grad()
    def generate(self, model, prompt, max_pred_tokens = 50, allowed_tokens = None,
        temp = 1.0, top_k = None, top_p = None, kv_cache = True,
        device = "cpu", stream_generate = False, no_sample = False, verbose= False):
        if device == 'cpu':
            print(f'⚠️ Running inference on CPU, consider setting `device="cuda"`!')
        
        model.eval()
        if prompt == '':
            try:
                star_indx = random.randint(0, self.n_vocab)
            except:
                star_indx = 1
            input_ids = torch.tensor([[star_indx]], dtype=torch.long).to(device)
        else:
            input_ids = torch.tensor([self.encode(prompt)], dtype=torch.long).to(device)
        start_pos = input_ids.shape[1]
        kv_caches = [dict() for _ in range(model.Nx_blocks)] if kv_cache else None

        x = input_ids
        gen = torch.Generator(device=device)

        for token_iter in range(max_pred_tokens):
            if kv_cache:
                logits = model._forward_with_cache(x[:, -1:], start_pos + token_iter, kv_caches)
            else:
                logits = model(x[:, -model.context_length:])

            logits = logits[:, -1, :] / temp

            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = -float("Inf")

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)

                mask = cumprobs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0

                indices_to_remove = mask.scatter(1, sorted_idx, mask)
                logits = logits.masked_fill(indices_to_remove, -float("Inf"))

            probs = F.softmax(logits, dim=-1)

            next_token = None
            if no_sample:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                for idx in sorted_idx[0]:
                    token_id = idx.item()
                    if allowed_tokens is None or token_id in allowed_tokens:
                        next_token = token_id
                        break
            else:
                token_id = torch.multinomial(probs, 1, generator=gen)
                
                if allowed_tokens is None or token_id in allowed_tokens:
                    next_token = token_id
                
            if next_token is None:
                next_token = self.special_tokens.get("<|endoftext|>")
                x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=-1)
                if stream_generate:
                    print(self.decode(next_token), end="", flush=True)
                break
            
            x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=-1)

            if stream_generate:
                print(self.decode(next_token[0][0].item()), end="", flush=True)

        out_tokens = x[0].tolist()
        out_text = self.decode(out_tokens)
        
        model.train()
        
        return out_text
    
    
    @torch.no_grad()
    def generate_batch(self, model, prompts, max_pred_tokens = 50, allowed_tokens = None,
        temp = 1.0, top_k = None, top_p = None, kv_cache = True,
        device = "cpu", no_sample = False):

        model.eval()

        encoded = [self.encode(p) for p in prompts]
        max_len = max(len(e) for e in encoded)
        batch_size = len(prompts)

        pad_token_id = self.special_tokens.get(ENDOFTEXT, ENDOFTEXT_IDX)
        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=pad_token_id,
            dtype=torch.long
        )

        for i, e in enumerate(encoded):
            input_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)

        input_ids = input_ids.to(device)
        start_pos = input_ids.shape[1]
        kv_caches = [dict() for _ in range(model.Nx_blocks)] if kv_cache else None

        x = input_ids
        gen = torch.Generator(device=device)

        for token_iter in range(max_pred_tokens):
            if kv_cache:
                logits = model._forward_with_cache(x[:, -1:], start_pos + token_iter, kv_caches)
            else:
                logits = model(x[:, -model.context_length:])

            logits = logits[:, -1, :] / temp

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = -float("Inf")

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask = cumprobs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                indices_to_remove = mask.scatter(1, sorted_idx, mask)
                logits = logits.masked_fill(indices_to_remove, -float("Inf"))

            probs = F.softmax(logits, dim=-1)

            if no_sample:
                next_tokens = probs.argmax(dim=-1)
            else:
                next_tokens = torch.multinomial(probs, 1, generator=gen).squeeze(1)

            if allowed_tokens is not None:
                filtered_tokens = []
                for t in next_tokens.tolist():
                    if t in allowed_tokens:
                        filtered_tokens.append(t)
                    else:
                        filtered_tokens.append(pad_token_id)
                next_tokens = torch.tensor(filtered_tokens, device=device, dtype=torch.long)

            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

        results = [self.decode(seq.tolist()) for seq in x]
        model.train()
        return results


    @torch.no_grad()
    def generate_batch_non_parallel(self, model, prompts, max_pred_tokens = 50, allowed_tokens = None,
        temp = 1.0, top_k = None, top_p = None, kv_cache = True,
        device = "cpu", no_sample = False):
        outputs = []
        for prompt in prompts:
            out_text = self.generate(
                model=model,
                prompt=prompt,
                max_pred_tokens=max_pred_tokens,
                allowed_tokens=allowed_tokens,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                kv_cache=kv_cache,
                device=device,
                stream_generate=False,
                no_sample=no_sample,
            )
            outputs.append(out_text)
        return outputs

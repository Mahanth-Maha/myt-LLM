import tiktoken

# Special tokens
ENDOFTEXT = "<|endoftext|>"
MAHAFILESEP = "<|maha_sep|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

def get_encoder():
    # enc = tiktoken.get_encoding("cl100k_base")
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.Encoding(
        name="cl100k_maha",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            MAHAFILESEP: 100264,
        }
    )
    return enc

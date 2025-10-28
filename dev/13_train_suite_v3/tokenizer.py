import tiktoken

ENDOFTEXT = "<|endoftext|>"
MAHAFILESEP = "<|maha_sep|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

ENDOFTEXT_IDX = 100257
MAHAFILESEP_IDX = 100264

spl_tok_dict = {
    MAHAFILESEP,
    ENDOFTEXT
}

def get_encoder():
    # enc = tiktoken.get_encoding("cl100k_base")
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.Encoding(
        name="cl100k_maha",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            MAHAFILESEP: MAHAFILESEP_IDX,
        }
    )
    return enc


def convert2hr(num):
    for unit in ['','K','M','B']:
        if abs(num) < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000.0
    return f"{num:.2f}T"
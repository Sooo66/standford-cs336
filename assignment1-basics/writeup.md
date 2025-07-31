## Q: Understanding Unicode
(a) chr(0) returns '\x00'
(b) print(chr(0)) returns empty; chr(0).__repr__() returns "'\\x00'"
(c) when occur in text, it's empty.

> __repr__() is for developer, always return a string. __str__() is for user, returns a string too.

Knowledge
- unicode standard: character <--> code point.
- unicode encoding: character <--> a sequence of bytes. such as: utf-8. utf-16, utf-32.

## Q: Unicode encodings
(a) The length of bytes encoded by utf-16/32 is longer than utf-8.
(b) utf-8 is a variable-lengeht encoding. bytes([x]) is different bytes(x), bytes(int) returns 0 bytes with length x.
(c) b'\xff\xff'; a two bytes seqence must begin with b'110xxxxx 10xxxxxx'

## Q: BPE Training on TinyStories.
(a) ~970s, ~460000KB; longest byte: [b' surpris', b' happily', b' quickly', b' magical', b' backyar']
(b) most cost part: pre_tokenize.

## Q: Experiments with tokenizers
(a) ts compression ration: 1.47 bytes/token; owt compression ration: 1.51 bytes/token.
(b) compression ration of tokenizing owt docs by ts tokenizer: 1.34 bytes/token.
(c) owt tokenization throughput: 668604.92 bytes/second. 892 * 1024**3 / 668604.92 = 1432501.73 seconds = 397.92 hours.
(d) vocab size if smaller than np.uint16 MAX_VALUE.

## Q: Transformer accounting
(a) hands calu: 2127057600, numel: 2127057600, 7.9239070415 GB to load(fp32).
(b) - Attn
    q(k, v)_proj: '... seq d_model, ... d_model d_model -> ... seq d_model'; calu: '2 * seq * d * d * 3'
    scale_attn: '... head seq d_k, ... head seq d_k -> ... head seq seq'; calu: '2 * head * seq * seq * d_k'
                '... head seq seq, ... head seq d_v -> ... head seq d_v'; calu: '2 * head * seq * seq * d_v'
    o_proj: '... seq d_model, ... d_model d_model -> ... seq d_model'; calu: '2 * seq * d * d'
    Total calu: 8 * seq * d * d + 4 * seq * seq * d = 27,682,406,400
    - FFN
    Gate: '... seq d_model, ... d_model d_ff'; calu: 'seq * d_model * d_ff'
    up: '... seq d_model, ... d_model d_ff'; calu: 'seq * d_model * d_ff'
    down: '... seq d_ff, ... d_ff d_model'; calu: 'seq * d_ff * d_model'
    Total calu: 3 * seq * d_model * d_ff = 31,457,280,000
    - lm_head
    '... seq d_model, ... d_model vocab_size -> ... seq vocab_size'; calu: 'seq * d_model * vocab_size' = 82,341,068,800

    Total FLOPs: num_layers * (4sd(2d+s) + (3s * d_model * d_ff)) + seq * d_model * vocab_size = 141,480,755,200
(c) lm_head is most expensive.
(d) - GPT-small:  96,636,764,160  + 181,193,932,800 + 39,523,713,024
    - GPT-medium: 309,237,645,312 + 161,061,273,600 + 52,698,284,032
    - GPT-large:  676,457,349,120 + 905,969,664,000 + 65,872,855,040
    The ffn contribution for FLOPs increase with the grow of model size.
(e) seq: 1024 -> 16384
    FLOPs:  1,328,755,507,200 +  1,509,949,440,000 +    82,341,068,800 =   2,921,046,016,000
        -> 98,569,499,443,200 + 24,159,191,040,000 + 1,317,457,100,800 = 124,046,147,584,000
    The attn contribution for FLOPs increase with the grow of seq_len.


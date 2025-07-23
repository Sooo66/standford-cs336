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
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



# XLnet-gen
XLNet for generating language.

** Work in progress **

### Number of generated tokens
Thanks to relative position encoding of transformer XL, it is possible to generate as many token as possible (subjected to memory constraints). Although there won't be direct connection between tokens more than `max_mem_length` distance apart, but they are still indirectly conditioned.
ating infinite 

### Sampling schemes
- [x] top-k sampling
- [x] top-p sampling
- [ ] beam search
- [ ] FSA decoding

### Context Length
Current implementation feeds in last max_mem_length tokens of the prompts, so the remaining earlier tokens are not conditioned upon. However, since the hidden states are subsequently cached, all the generated tokens are conditioned upon both the max_mem_length tokens of the prompts and the subsequent generated tokens.

Todo: allow arbitrary length context by using sliding window approach.
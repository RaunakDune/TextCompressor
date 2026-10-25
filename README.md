# TextCompressor

Text compression with Context Tree Weighting, and its modern successors.

The original 2018 version (`textCompressor.py`, `compressorTest.py`, kept for
history) estimated the entropy of `enwik4` with a fixed-order bit lookup table
and KT estimator; its CTW tree was left unfinished. **`ctwmix.py`** is the
upgrade: a real compressor/decompressor (lossless, verified round-trip) that
finishes the CTW idea and then layers on the techniques that superseded it.

```
pip install -r requirements.txt          # numpy, numba
python ctwmix.py c enwik4 enwik4.cx      # compress   (default --mode cm)
python ctwmix.py d enwik4.cx enwik4.out  # decompress
python ctwmix.py bench enwik4            # all modes + round-trip check
```

Modes: `cm` (full model, default), `cts` (pure Context Tree Switching),
`ctw` (pure Context Tree Weighting), `cm-noctw` (mixing without the tree).

## Results

Compressed bytes (smaller is better). enwik*N* = first 10^*N* bytes of
`enwik8` (md5 `a1fa5ffd…bbb36a`). ctwmix sizes include its 9-byte header and
were all decoded back and compared byte-for-byte.

| Compressor | enwik4 (10 KB) | enwik5 (100 KB) | enwik6 (1 MB) |
|---|---:|---:|---:|
| Original repo, best order (17 bits)\* | 4,977 | – | – |
| zstd -22 | 3,665 | 33,740 | 300,074 |
| xz -9e | 3,656 | 33,016 | 290,732 |
| bzip2 -9 | 3,733 | 31,372 | 281,323 |
| PPMd (best of orders 4–16) | 3,217 | 27,769 | 245,509 |
| ctwmix `--mode ctw` | 3,484 | 29,768 | 272,863 |
| ctwmix `--mode cts` | 3,399 | 29,170 | 265,587 |
| **ctwmix `--mode cm`** | **2,972** | **25,595** | **217,663** |
| paq8px v217 -5 (reference) | 2,725 | 23,615 | 195,543 |

\* ideal code length only — no file was produced.

On enwik6 the `cm` mode is 1.74 bpc: 25% smaller than xz, 11% smaller than
PPMd, 18% smaller than pure CTS. paq8px (a C++ model with hundreds of
contexts and text/XML-specific models) is still ~11% smaller but ~5× slower.
Speed here: ~55 s/MB compress or decompress (Python + numba, one core);
`cts` runs at ~10 s/MB.

## What changed, and where it came from

**1. A finished, byte-oriented CTW.** The 2018 tree worked on raw bit
history, so an order-16 bit context spanned two bytes with no notion of
byte alignment. The new tree uses the standard decomposition for ASCII: the
context is the previous *D* bytes plus the partial byte, with 255 binary
nodes per byte context (Willems, Shtarkov & Tjalkens 1995). Weighting is
done in fixed-point log domain:

    pw(s) = w·pe(s) + (1−w)·pw(child),   w = Pe(s) / (Pe(s) + Π Pw(children))

so encoder and decoder agree bit-for-bit on every platform. Two
long-known tweaks for text: KT α = 1/16 instead of 1/2 (contexts in text
are nearly deterministic) and count halving at 255 for non-stationarity.

**2. Context Tree Switching.** CTS (Veness, Ng, Hutter & Bowling, DCC
2012) keeps CTW's tree but lets each node *switch* between "stop here" and
"go deeper" over time, at the same cost. Here it gives a 2–3% gain over
CTW at every size, matching the paper's finding that it consistently beats
CTW on the Calgary Corpus.

**3. Context mixing (PAQ → GLN).** The large gain comes from not trusting
any single model. 16 context models (orders 0–6, words, word bigrams,
skip-word, sparse, column/indent, two indirect, case-folded order-5),
a run model, a match model and the CTS tree each emit stretched
probabilities, and a 2-layer gated logistic mixer combines them. Each
first-layer neuron keeps one weight vector per value of its gating context
and learns locally on its own log loss — PAQ's mixer, which Veness et al.
later formalised as **Gated Linear Networks** (AAAI 2021). Each context slot
stores both a direct adaptive probability and a small non-stationary
bit-history state that a learned StateMap turns into a probability
(PAQ-style indirect modelling).

**4. SSE.** Three adaptive probability maps (order-0, order-1, hashed
order-3) refine the mixer output.

**5. An actual arithmetic coder.** 32-bit carry-less binary coder.

### What was tried and not kept

- An LSTM input (as in cmix / paq8px `-L`). paq8px with LSTM only went
  from 23,615 to 23,535 bytes on enwik5 at ~70 s per 100 KB; at these
  file sizes it isn't worth it in Python.
- A fast second counter per context (+0.13%), deterministic-run inputs
  (no gain), bigger hash tables (+0.16% on enwik6).

### Where the field is now

The Hutter Prize record on enwik9 is fx2-cmix (Kaido Orav & Byron Knoll,
Sept 2024) at 110,793,128 bytes including the decompressor. On Matt
Mahoney's Large Text Compression Benchmark the top entries are now
transformer/context-mixing hybrids around 92 MB on enwik9; nncp v3.2
(online Transformer) is at 106.6 MB, paq8px_v206 at 124.7 MB, and the one
CTW entry at 212 MB. Delétang et al. (ICLR 2024) make the general point
that a predictor is a compressor and vice versa — which is exactly the
structure here: everything above step 5 is a predictor.

## Possible next steps

- Port the predictor to C (the numba code is already integer-only and
  maps 1:1) for a ~10× speedup.
- Skip-CTS (Bellemare, Veness & Talvitie, ICML 2014) for sparse contexts.
- A text preprocessor (capital flags, dictionary) as the Hutter Prize
  entries use.

## References

- F. Willems, Y. Shtarkov, T. Tjalkens, *The Context-Tree Weighting Method: Basic Properties*, IEEE Trans. IT, 1995.
- J. Veness, K. S. Ng, M. Hutter, M. Bowling, [*Context Tree Switching*](https://arxiv.org/abs/1111.3182), DCC 2012.
- M. Bellemare, J. Veness, E. Talvitie, [*Skip Context Tree Switching*](https://www.academia.edu/7812162/Skip_Context_Tree_Switching), ICML 2014.
- J. Veness et al., [*Gated Linear Networks*](https://arxiv.org/abs/1910.01526), AAAI 2021.
- G. Delétang et al., [*Language Modeling Is Compression*](https://arxiv.org/abs/2309.10668), ICLR 2024.
- M. Mahoney, [*Large Text Compression Benchmark*](https://mattmahoney.net/dc/text.html).
- [Hutter Prize](https://en.wikipedia.org/wiki/Hutter_Prize) winners.
- [paq8px](https://github.com/hxim/paq8px) (reference numbers above, v217).

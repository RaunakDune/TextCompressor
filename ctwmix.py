"""
ctwmix -- a lossless text compressor built on Context Tree Weighting,
upgraded with the context-mixing ideas that superseded it.

    python ctwmix.py c  input  output   [--mode cm|cts|ctw|cm-noctw]
    python ctwmix.py d  input  output
    python ctwmix.py bench file [file ...]

Bytes are coded MSB first, one bit at a time; every model sees the partial
byte c0 (1..255) as part of its context.

  1. Predictors
     * CTW/CTS   byte-oriented Context Tree Weighting (Willems, Shtarkov &
                 Tjalkens 1995) with the 255-node binary decomposition per
                 byte context, depth D bytes, KT estimator with tunable alpha
                 (alpha < 1/2 suits the near-deterministic contexts of text)
                 and count halving for non-stationarity. Weighting is done in
                 fixed-point log domain, so encoder and decoder are bit-exact
                 on any platform.  Context Tree Switching (Veness, Ng, Hutter
                 & Bowling 2012) is the default tree inside cm; it replaces
                 CTW's fixed 1/2-1/2 weighting with switching weights.
     * orders 0-6, word, word-bigram and sparse context models. Each hashed,
       checksummed slot holds a direct adaptive probability *and* a small
       non-stationary bit-history state that is mapped to a probability by
       a learned StateMap (PAQ's "indirect" modelling).
     * match model for long repeats (lpaq / paq8 style).
  2. Mixing      2-layer gated logistic mixer.  Each first-layer neuron owns
                 one weight vector per value of its gating context and learns
                 locally on its own log loss -- PAQ's mixer, which Veness et
                 al. formalised as a Gated Linear Network (GLN).
  3. SSE / APM   two adaptive probability maps refine the mixed estimate.
  4. Coding      32-bit carry-less binary arithmetic coder, 12-bit probs.

The original repo only *estimated* entropy with a fixed-order bit table;
this produces a real file and decodes it back.
"""
import sys, time, struct, math
import numpy as np
from numba import njit

MAGIC = b"CTWX"
MODES = {"cm": 0, "ctw": 1, "cm-noctw": 2, "cts": 3}

# ---------------------------------------------------------------- tables ---
# Logistic domain ("stretch" units): 256 per nat, clipped to +-2047.
_SQ = np.zeros(4096, np.int64)
for _i in range(4096):
    _SQ[_i] = int(4096.0 / (1.0 + math.exp(-(_i - 2048) / 256.0)))
_SQ = np.clip(_SQ, 1, 4095)
_ST = np.zeros(4096, np.int64)            # stretch = inverse of squash
_pi = 0
for _x in range(-2047, 2048):
    _v = int(_SQ[_x + 2048])
    for _j in range(_pi, _v + 1):
        _ST[_j] = _x
    _pi = _v + 1
for _j in range(_pi, 4096):
    _ST[_j] = 2047
# ln(p/4096) in 1/256 nat, for CTW's log-domain bookkeeping
_LN = np.zeros(4096, np.int64)
for _i in range(1, 4096):
    _LN[_i] = int(round(256.0 * math.log(_i / 4096.0)))
_LN[0] = _LN[1]
# softplus for log-add: 256*ln(1+exp(-k/256))
_SP = np.zeros(4097, np.int64)
for _k in range(4097):
    _SP[_k] = int(round(256.0 * math.log1p(math.exp(-_k / 256.0))))
LN2 = int(round(256.0 * math.log(2.0)))
# reciprocal table for adaptive-rate counters: 65536/(n+1.5)
_DT = np.array([int(65536 / (n + 1.5)) for n in range(1024)], np.int64)

# Non-stationary bit-history states: (n0, n1, last bit), n0,n1 in 0..HC-1.
# state = (n0*HC + n1)*2 + last.  Next-state tables built once here.
HC = 16
NS = HC * HC * 2
_NXT = np.zeros((NS, 2), np.int64)
_SINIT = np.zeros(NS, np.int64)
for _s in range(NS):
    _n0, _n1 = (_s >> 1) // HC, (_s >> 1) % HC
    for _y in (0, 1):
        a, b = _n0, _n1
        if _y:
            b = min(b + 1, HC - 1)
            if a > 2:
                a = a // 2 + 1
        else:
            a = min(a + 1, HC - 1)
            if b > 2:
                b = b // 2 + 1
        _NXT[_s, _y] = (a * HC + b) * 2 + _y
    _SINIT[_s] = int(65536 * (_n1 + 0.4) / (_n0 + _n1 + 0.8))


@njit(cache=True, inline="always")
def squash(d):
    if d > 2047:
        d = 2047
    if d < -2047:
        d = -2047
    return _SQ[d + 2048]


@njit(cache=True, inline="always")
def clip12(p):
    if p < 1:
        return 1
    if p > 4095:
        return 4095
    return p


@njit(cache=True, inline="always")
def logadd(a, b):
    if a > b:
        k = a - b
        return a + (_SP[k] if k < 4096 else 0)
    k = b - a
    return b + (_SP[k] if k < 4096 else 0)


@njit(cache=True, inline="always")
def hmix(x):
    x = (x * 0x9E3779B1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 13
    return x


# ------------------------------------------------------------- the codec ---
@njit(cache=True)
def codec(src, nbytes, decode, mode, P):
    """Encode src[0:nbytes] (decode=0) or decode src into nbytes bytes
    (decode=1).  Returns (output array, ideal code length in bits)."""
    CTW_D = P[0]          # CTW depth in bytes
    CTW_A = P[1]          # KT alpha in 1/16 units (8 = classic KT 1/2)
    CTW_LIM = P[2]        # CTW count halving threshold
    CTW_BITS = P[3]       # log2 CTW node table size
    CM_BITS = P[4]        # log2 slots per context-model table
    LIMIT = P[5]          # direct-counter limit (higher = more stationary)
    LR0 = P[6]            # initial mixer learning rate
    LRMIN = P[7]          # floor mixer learning rate
    MINLEN = P[8]         # match model min length
    LRDEC = P[9]          # learning-rate half-life in bytes
    APMW = P[10]          # weight (of 4) given to the APM outputs
    TREE = P[11]          # tree model inside cm: 0 = CTW, 1 = CTS
    CTS_K = P[12]         # CTS switching rate alpha_n = 1/(n + K)

    use_ctw = mode != 2
    use_cm = mode == 0 or mode == 2
    if mode == 1:
        TREE = 0
    if mode == 3:
        TREE = 1

    # ---------- buffers & arithmetic coder state
    if decode:
        out = np.zeros(nbytes, np.uint8)
        buf = out
    else:
        out = np.zeros(nbytes + nbytes // 2 + 1024, np.uint8)
        buf = src
    inp = src
    op = 0
    ip = 0
    x1 = 0
    x2 = 0xFFFFFFFF
    x = 0
    if decode:
        for _ in range(4):
            x = (x << 8) | (inp[ip] if ip < inp.shape[0] else 0)
            ip += 1

    # ---------- CTW node table (hashed, checksummed)
    csz = (1 << CTW_BITS) if use_ctw else 1
    cmask = csz - 1
    c_n0 = np.zeros(csz, np.int32)
    c_n1 = np.zeros(csz, np.int32)
    c_lpe = np.zeros(csz, np.int64)     # log Pe(s)
    c_lk = np.zeros(csz, np.int64)      # sum of log Pw(children)
    c_lpw = np.zeros(csz, np.int64)     # log Pw(s)
    c_chk = np.zeros(csz, np.int32)
    c_idx = np.zeros(CTW_D + 1, np.int64)
    c_pe = np.zeros(CTW_D + 1, np.int64)
    c_pw = np.zeros(CTW_D + 1, np.int64)
    ctxh = np.zeros(CTW_D + 1, np.int64)

    # ---------- context models
    # 0 order0, 1 order1, 2..6 order2..6, 7 word, 8 word bigram,
    # 9 sparse (bytes 2-3 back), 10 c1 + word length, 11 column/indent
    # (byte above in previous line), 12-13 indirect (what followed the
    # order-1/order-2 context last time), 14 skip-word bigram,
    # 15 case-folded order-5
    NCM = 16
    tsz = (1 << CM_BITS) if use_cm else 1
    tmask = tsz - 1
    t_p = np.full((NCM, tsz), 32768, np.int32)   # direct probability
    t_n = np.zeros((NCM, tsz), np.int16)         # count
    t_s = np.zeros((NCM, tsz), np.int16)         # bit-history state
    t_c = np.zeros((NCM, tsz), np.int16)         # checksum
    SM = np.zeros((NCM, NS), np.int64)           # state -> prob
    SMn = np.zeros((NCM, NS), np.int64)
    for m in range(NCM):
        for s in range(NS):
            SM[m, s] = _SINIT[s]
    h = np.zeros(NCM, np.int64)
    slot = np.zeros(NCM, np.int64)
    sst = np.zeros(NCM, np.int64)

    # ---------- match model
    MMB = 22
    mm_tab = np.zeros(1 << MMB, np.int64)
    mm_ptr = 0
    mm_len = 0
    MML = 63
    mm_sm = np.full((MML + 1) * 2, 32768, np.int64)
    mm_smn = np.zeros((MML + 1) * 2, np.int64)
    mm_ctx = -1
    mm_bit = 0

    # ---------- run model: per byte-context, the byte that followed it last
    #            time and how often in a row (orders 1..6 + word)
    NRM = 7
    r_b = np.zeros((NRM, tsz), np.int32)
    r_n = np.zeros((NRM, tsz), np.int32)
    r_c = np.zeros((NRM, tsz), np.int32)
    r_ix = np.zeros(NRM, np.int64)
    r_ctx = np.zeros(NRM, np.int64)
    RSM = np.full((NRM, 64), 32768, np.int64)
    RSMn = np.zeros((NRM, 64), np.int64)

    # ---------- mixer: 3 inputs per CM + run + match 2 + ctw 2 + bias
    NIN = NCM * 3 + NRM + 2 + 2 + 1
    NSEL = 7
    W = np.zeros((NSEL, 2048, NIN), np.int64)
    for s in range(NSEL):
        for k in range(2048):
            for i in range(NIN):
                W[s, k, i] = 3000
    W2 = np.zeros((256, NSEL), np.int64)
    for k in range(256):
        for s in range(NSEL):
            W2[k, s] = 65536 // NSEL
    xin = np.zeros(NIN, np.int64)
    sel = np.zeros(NSEL, np.int64)
    s1 = np.zeros(NSEL, np.int64)
    p1 = np.zeros(NSEL, np.int64)

    # ---------- APMs (33 interpolation buckets each)
    A1 = np.zeros(256 * 33, np.int64)
    A2 = np.zeros(65536 * 33, np.int64)
    A3 = np.zeros(65536 * 33, np.int64)
    a3i = 0
    for j in range(33):
        v = squash((j - 16) * 128) * 16
        for k in range(256):
            A1[k * 33 + j] = v
        for k in range(65536):
            A2[k * 33 + j] = v
            A3[k * 33 + j] = v

    c4 = 0
    word = 0
    word1 = 0
    word2 = 0
    wlen = 0
    col = 0
    lstart = 0      # start of current line
    pstart = 0      # start of previous line
    ind1 = np.zeros(256, np.int64)
    ind2 = np.zeros(65536, np.int64)
    lr = LR0
    tot = 0.0
    p = 2048
    pm = 2048
    a1i = 0
    a2i = 0
    wt = 0
    hits = 0

    for pos in range(nbytes):
        # ================= byte-boundary context update
        c1 = c4 & 0xFF
        if use_ctw:
            hh = 0
            for d in range(CTW_D + 1):
                ctxh[d] = hh
                if d < CTW_D:
                    b = buf[pos - 1 - d] if pos - 1 - d >= 0 else 0
                    hh = hmix(hh * 773 + b + 1 + (d << 9))
        if use_cm:
            h[0] = 0
            h[1] = c1 << 8
            hh = 0
            for o in range(1, 7):
                b = buf[pos - o] if pos - o >= 0 else 0
                hh = hmix(hh * 271 + b + 1 + o * 0x1000000)
                if o >= 2:
                    h[o] = hh
            h[7] = hmix(word * 31 + 7)
            h[8] = hmix((word + word1 * 0x3F1) * 37 + 8)
            h[9] = hmix(((c4 >> 8) & 0xFFFF) * 41 + 9)
            h[10] = hmix((c1 | (min(wlen, 15) << 8)) * 43 + 10)
            above = buf[pstart + col] if pstart + col < lstart else 0
            h[11] = hmix(((min(col, 255) << 8) | above) * 47 + 11)
            h[12] = hmix((c1 | (ind1[c1] << 8)) * 53 + 12)
            c2 = c4 & 0xFFFF
            h[13] = hmix((c2 | (ind2[c2] << 16)) * 59 + 13)
            h[14] = hmix((word + word2 * 0x5A3) * 61 + 14)
            hh = 15
            for o in range(1, 6):
                b = buf[pos - o] if pos - o >= 0 else 0
                if b >= 65 and b <= 90:
                    b += 32
                hh = hmix(hh * 277 + b + 1)
            h[15] = hh
            for r in range(NRM):
                z = hmix((h[r + 1] if r < 6 else h[7]) + 0x7777 * (r + 1))
                ix = z & tmask
                ck = (z >> 20) | 1
                if r_c[r, ix] != ck:
                    r_c[r, ix] = ck
                    r_n[r, ix] = 0
                r_ix[r] = ix
            # match model: extend or look up previous occurrence
            if mm_len > 0 and buf[mm_ptr] == c1:
                mm_len += 1
                mm_ptr += 1
                if mm_len > MML:
                    mm_len = MML
            else:
                mm_len = 0
            if pos >= MINLEN:
                hm = 0
                for k in range(MINLEN):
                    hm = (hm * 0x2F0F1 + buf[pos - 1 - k] + 1) & 0xFFFFFFFF
                hm = hmix(hm) & ((1 << MMB) - 1)
                if mm_len == 0:
                    cand = mm_tab[hm]
                    if cand > 0:
                        L = 0
                        while L < MML and cand - 1 - L >= 0 and buf[cand - 1 - L] == buf[pos - 1 - L]:
                            L += 1
                        if L >= MINLEN:
                            mm_len = L
                            mm_ptr = cand
                mm_tab[hm] = pos

        c0 = 1
        for bit_i in range(8):
            # ================= predict
            ni = 0
            if use_cm:
                hits = 0
                for m in range(NCM):
                    if m == 0:
                        ix = c0
                    elif m == 1:
                        ix = (h[1] | c0) & tmask
                    else:
                        z = hmix(h[m] ^ ((c0 * 0x5BD1E995) & 0xFFFFFFFF))
                        ix = z & tmask
                        ck = (z >> 20) | 1
                        if t_c[m, ix] != ck:
                            t_c[m, ix] = ck
                            t_p[m, ix] = 32768
                            t_n[m, ix] = 0
                            t_s[m, ix] = 0
                    slot[m] = ix
                    n = t_n[m, ix]
                    s = t_s[m, ix]
                    sst[m] = s
                    if n == 0:
                        xin[ni] = 0
                        xin[ni + 1] = 0
                        xin[ni + 2] = 0
                    else:
                        if m >= 2 and m <= 6:
                            hits += 1
                        pd = t_p[m, ix] >> 4
                        xin[ni] = _ST[pd]
                        xin[ni + 1] = _ST[SM[m, s] >> 4]
                        xin[ni + 2] = (pd - 2048) >> 1
                    ni += 3
                # run model
                for r in range(NRM):
                    ix = r_ix[r]
                    rn = r_n[r, ix]
                    r_ctx[r] = -1
                    xin[ni] = 0
                    if rn > 0:
                        rb = r_b[r, ix] | 256
                        if (rb >> (8 - bit_i)) == c0:
                            eb = (rb >> (7 - bit_i)) & 1
                            q = (rn if rn < 16 else 15 + (rn >> 4)) 
                            if q > 31:
                                q = 31
                            r_ctx[r] = q * 2 + eb
                            xin[ni] = _ST[RSM[r, q * 2 + eb] >> 4]
                    ni += 1
                # match model
                if mm_len > 0:
                    eb = (buf[mm_ptr] >> (7 - bit_i)) & 1
                    mm_bit = eb
                    mm_ctx = mm_len * 2 + eb
                    xin[ni] = _ST[mm_sm[mm_ctx] >> 4]
                    v = mm_len * 64
                    if v > 2047:
                        v = 2047
                    xin[ni + 1] = v if eb else -v
                else:
                    mm_ctx = -1
                    xin[ni] = 0
                    xin[ni + 1] = 0
                ni += 2

            # --- CTW: pw(s) = w*pe(s) + (1-w)*pw(child), w = Pe/(Pe + K*Pw)
            pctw = 2048
            if use_ctw:
                for d in range(CTW_D + 1):
                    z = hmix(ctxh[d] ^ ((c0 * 0x9E3779B1 + d * 0x632BE5AB) & 0xFFFFFFFF))
                    ix = z & cmask
                    ck = (z >> 16) | 0x10000
                    if c_chk[ix] != ck:
                        c_chk[ix] = ck
                        c_n0[ix] = 0
                        c_n1[ix] = 0
                        c_lpe[ix] = 0
                        c_lk[ix] = 0
                        c_lpw[ix] = 0
                    c_idx[d] = ix
                    n0 = c_n0[ix]
                    n1 = c_n1[ix]
                    c_pe[d] = clip12(((16 * n1 + CTW_A) * 4096) // (16 * (n0 + n1) + 2 * CTW_A))
                pw = c_pe[CTW_D] * 16
                c_pw[CTW_D] = c_pe[CTW_D]
                for d in range(CTW_D - 1, -1, -1):
                    ix = c_idx[d]
                    if TREE == 0:
                        w = squash(c_lpe[ix] - c_lk[ix])
                    else:
                        w = squash(c_lk[ix])    # CTS: log(w_stop / w_split)
                    pw = (w * c_pe[d] * 16 + (4096 - w) * pw) >> 12
                    c_pw[d] = clip12(pw >> 4)
                pctw = clip12(pw >> 4)
                if use_cm:
                    xin[ni] = _ST[pctw]
                    xin[ni + 1] = (pctw - 2048) >> 1
                    ni += 2

            if use_cm:
                xin[ni] = 256      # bias
                ni += 1
                # --- layer 1: NSEL gated neurons
                sel[0] = c0
                if mm_len > 0:
                    ml = mm_len if mm_len < 16 else 16 + (mm_len >> 3)
                    sel[1] = (ml * 2 + mm_bit) * 32 + bit_i * 4 + (hits >> 1) + 32
                else:
                    sel[1] = bit_i * 4 + (hits >> 1)
                sel[2] = hits * 256 + c0
                sel[3] = c1 * 8 + bit_i
                sel[4] = 0
                sel[5] = ((c4 >> 8) & 0xFF) * 8 + bit_i
                sel[6] = (min(wlen, 7) * 8 + min(col, 7)) * 8 + bit_i
                for s in range(NSEL):
                    k = sel[s]
                    dot = 0
                    for i in range(ni):
                        dot += W[s, k, i] * xin[i]
                    dot >>= 16
                    if dot > 2047:
                        dot = 2047
                    if dot < -2047:
                        dot = -2047
                    s1[s] = dot
                    p1[s] = squash(dot)
                # --- layer 2
                dot = 0
                for s in range(NSEL):
                    dot += W2[c0, s] * s1[s]
                dot >>= 16
                if dot > 2047:
                    dot = 2047
                if dot < -2047:
                    dot = -2047
                pm = squash(dot)
                # --- APM / SSE refinement
                stp = dot + 2048
                lo = stp >> 7
                wt = stp & 127
                a1i = c0 * 33 + lo
                a2i = ((c0 | (c1 << 8)) & 0xFFFF) * 33 + lo
                pa1 = (A1[a1i] * (128 - wt) + A1[a1i + 1] * wt) >> 11
                pa2 = (A2[a2i] * (128 - wt) + A2[a2i + 1] * wt) >> 11
                a3i = (hmix((c4 & 0xFFFFFF) * 0x10000 + c0) & 0xFFFF) * 33 + lo
                pa3 = (A3[a3i] * (128 - wt) + A3[a3i + 1] * wt) >> 11
                p = clip12((pm * (4 - APMW) * 3 + (pa1 + pa2 + pa3) * APMW) // 12)
            else:
                p = pctw

            # ================= arithmetic-code the bit
            xmid = x1 + ((x2 - x1) >> 12) * p + ((((x2 - x1) & 0xFFF) * p) >> 12)
            if decode:
                y = 1 if x <= xmid else 0
            else:
                y = (buf[pos] >> (7 - bit_i)) & 1
            if y:
                x2 = xmid
            else:
                x1 = xmid + 1
            while ((x1 ^ x2) & 0xFF000000) == 0:
                if decode:
                    x = ((x << 8) & 0xFFFFFFFF) | (inp[ip] if ip < inp.shape[0] else 0)
                    ip += 1
                else:
                    out[op] = x2 >> 24
                    op += 1
                x1 = (x1 << 8) & 0xFFFFFFFF
                x2 = ((x2 << 8) & 0xFFFFFFFF) | 255
            tot -= math.log2(p / 4096.0 if y else 1.0 - p / 4096.0)

            # ================= update
            if use_ctw and TREE == 1:
                # Context Tree Switching (Veness, Ng, Hutter & Bowling 2012):
                # two weights per node, mixed with switching rate alpha_n
                for d in range(CTW_D + 1):
                    ix = c_idx[d]
                    pe = c_pe[d]
                    if d < CTW_D:
                        n = c_lpw[ix]
                        al = 4096 // (n + CTS_K)
                        if al < 1:
                            al = 1
                        if al > 2048:
                            al = 2048
                        la = _LN[al]
                        l1a = _LN[4096 - al]
                        pc = c_pw[d + 1]
                        a = c_lk[ix] + _LN[pe if y else 4096 - pe]
                        b = _LN[pc if y else 4096 - pc]
                        ns = logadd(l1a + a, la + b)
                        nc = logadd(l1a + b, la + a)
                        dl = ns - nc
                        if dl > 8192:
                            dl = 8192
                        if dl < -8192:
                            dl = -8192
                        c_lk[ix] = dl
                        c_lpw[ix] = n + 1
                    if y:
                        c_n1[ix] += 1
                    else:
                        c_n0[ix] += 1
                    if c_n0[ix] + c_n1[ix] > CTW_LIM:
                        c_n0[ix] = (c_n0[ix] + 1) >> 1
                        c_n1[ix] = (c_n1[ix] + 1) >> 1
            elif use_ctw:
                for d in range(CTW_D, -1, -1):
                    ix = c_idx[d]
                    pe = c_pe[d]
                    c_lpe[ix] += _LN[pe if y else 4096 - pe]
                    if y:
                        c_n1[ix] += 1
                    else:
                        c_n0[ix] += 1
                    if c_n0[ix] + c_n1[ix] > CTW_LIM:
                        c_n0[ix] = (c_n0[ix] + 1) >> 1
                        c_n1[ix] = (c_n1[ix] + 1) >> 1
                    old = c_lpw[ix]
                    if d == CTW_D:
                        nw = c_lpe[ix]
                    else:
                        nw = logadd(c_lpe[ix], c_lk[ix]) - LN2
                    c_lpw[ix] = nw
                    if d > 0:
                        c_lk[c_idx[d - 1]] += nw - old
            if use_cm:
                for m in range(NCM):
                    ix = slot[m]
                    n = t_n[m, ix]
                    t_p[m, ix] += (((y << 16) - t_p[m, ix]) * _DT[n]) >> 16
                    if n < LIMIT:
                        t_n[m, ix] = n + 1
                    s = sst[m]
                    if n > 0:
                        sn = SMn[m, s]
                        SM[m, s] += (((y << 16) - SM[m, s]) * _DT[sn]) >> 16
                        if sn < 1000:
                            SMn[m, s] = sn + 1
                    t_s[m, ix] = _NXT[s, y]
                for r in range(NRM):
                    q = r_ctx[r]
                    if q >= 0:
                        n = RSMn[r, q]
                        RSM[r, q] += (((y << 16) - RSM[r, q]) * _DT[n]) >> 16
                        if n < 1000:
                            RSMn[r, q] = n + 1
                if mm_ctx >= 0:
                    n = mm_smn[mm_ctx]
                    mm_sm[mm_ctx] += (((y << 16) - mm_sm[mm_ctx]) * _DT[n]) >> 16
                    if n < 1000:
                        mm_smn[mm_ctx] = n + 1
                    if mm_bit != y:
                        mm_len = 0
                # layer 1: each neuron learns locally on its own loss (GLN rule)
                for s in range(NSEL):
                    e = ((y << 12) - p1[s]) * lr
                    k = sel[s]
                    for i in range(ni):
                        W[s, k, i] += (xin[i] * e) >> 16
                e = ((y << 12) - pm) * lr
                for s in range(NSEL):
                    W2[c0, s] += (s1[s] * e) >> 16
                tgt = (y << 16) + (y << 5) - y - y
                A1[a1i] += ((tgt - A1[a1i]) * (128 - wt)) >> 13
                A1[a1i + 1] += ((tgt - A1[a1i + 1]) * wt) >> 13
                A2[a2i] += ((tgt - A2[a2i]) * (128 - wt)) >> 13
                A2[a2i + 1] += ((tgt - A2[a2i + 1]) * wt) >> 13
                A3[a3i] += ((tgt - A3[a3i]) * (128 - wt)) >> 13
                A3[a3i + 1] += ((tgt - A3[a3i + 1]) * wt) >> 13
            c0 = (c0 << 1) | y
        if decode:
            out[pos] = c0 & 0xFF

        # ================= byte done
        cb = c0 & 0xFF
        if use_cm:
            for r in range(NRM):
                ix = r_ix[r]
                if r_n[r, ix] > 0 and r_b[r, ix] == cb:
                    if r_n[r, ix] < 255:
                        r_n[r, ix] += 1
                else:
                    r_b[r, ix] = cb
                    r_n[r, ix] = 1
        ind1[c4 & 0xFF] = ((ind1[c4 & 0xFF] << 8) | cb) & 0xFFFF
        ind2[c4 & 0xFFFF] = ((ind2[c4 & 0xFFFF] << 8) | cb) & 0xFFFF
        c4 = ((c4 << 8) | cb) & 0xFFFFFFFF
        if cb == 10:
            pstart = lstart
            lstart = pos + 1
            col = 0
        else:
            col += 1
        if (cb >= 97 and cb <= 122) or (cb >= 65 and cb <= 90):
            if cb <= 90:
                cb += 32
            word = hmix(word * 263 + cb)
            wlen += 1
        else:
            if word != 0:
                word2 = word1
                word1 = word
            word = 0
            wlen = 0
        lr = LR0 * LRDEC // (LRDEC + pos)
        if lr < LRMIN:
            lr = LRMIN

    if not decode:
        for _ in range(4):
            out[op] = x1 >> 24
            op += 1
            x1 = (x1 << 8) & 0xFFFFFFFF
        return out[:op], tot
    return out, tot


PARAM_NAMES = ["D", "A", "LIM", "CB", "TB", "LIMIT", "LR0", "LRMIN", "MIN",
               "LRDEC", "APMW", "TREE", "K"]
DEFAULT_P = np.array([
    6,       # D      CTW depth (bytes)
    1,       # A      KT alpha in 1/16 (1 => alpha=1/16)
    255,     # LIM    CTW count halving limit
    22,      # CB     CTW table bits
    22,      # TB     context-model table bits
    255,     # LIMIT  direct-counter limit
    24,      # LR0    initial mixer learning rate
    6,       # LRMIN  floor mixer learning rate
    6,       # MIN    match model min length
    20000,   # LRDEC  learning-rate decay scale (bytes)
    2,       # APMW   APM share (of 4)
    1,       # TREE   tree model used inside cm: 0 CTW, 1 CTS
    128,     # K      CTS switching rate 1/(n+K)
], np.int64)


def compress(data: bytes, mode="cm", P=DEFAULT_P):
    arr = np.frombuffer(data, np.uint8)
    body, bits = codec(arr, len(arr), 0, MODES[mode], P)
    hdr = MAGIC + struct.pack("<BI", MODES[mode], len(arr))
    return hdr + body.tobytes(), bits


def decompress(blob: bytes, P=DEFAULT_P):
    assert blob[:4] == MAGIC, "not a ctwmix file"
    mode, n = struct.unpack("<BI", blob[4:9])
    arr = np.frombuffer(blob[9:], np.uint8)
    out, _ = codec(arr, n, 1, mode, P)
    return out.tobytes()


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    cmd = argv[1]
    mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "cm"
    if cmd == "c":
        data = open(argv[2], "rb").read()
        t = time.time()
        blob, bits = compress(data, mode)
        open(argv[3], "wb").write(blob)
        print(f"{len(data)} -> {len(blob)} bytes  "
              f"({8*len(blob)/max(1,len(data)):.3f} bpc, {time.time()-t:.1f}s)")
    elif cmd == "d":
        blob = open(argv[2], "rb").read()
        t = time.time()
        data = decompress(blob)
        open(argv[3], "wb").write(data)
        print(f"{len(blob)} -> {len(data)} bytes ({time.time()-t:.1f}s)")
    elif cmd == "bench":
        for f in [a for a in argv[2:] if not a.startswith("--")]:
            data = open(f, "rb").read()
            for m in ("ctw", "cts", "cm-noctw", "cm"):
                t = time.time()
                blob, bits = compress(data, m)
                tc = time.time() - t
                t = time.time()
                ok = decompress(blob) == data
                td = time.time() - t
                print(f"{f:10s} {m:9s} {len(data):>9d} -> {len(blob):>8d}  "
                      f"{8*len(blob)/len(data):.3f} bpc  "
                      f"roundtrip={'OK' if ok else 'FAIL'}  c {tc:.1f}s d {td:.1f}s",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

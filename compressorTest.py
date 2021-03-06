# enwik4
# Original Size:    10000
# 7 Zipped:         3719

from collections import defaultdict
import pandas as pd
from ggplot import *
import math

enw = open("enwik4", "rb").read()
NUMBER_OF_BITS = 16

def bitgen(x):
    for c in x:
        for i in range(8):
            yield int((c & (0x80>>i)) != 0)

def test(NUMBER_OF_BITS):
    bg = bitgen(enw)
    lookup = defaultdict(lambda: [0.5,1.0])
    HH = 0.0
    try:
        prevx = [-1] * NUMBER_OF_BITS
        while 1:
            x = next(bg)

            #Use lookup table
            px = tuple(prevx)        

            # lookup[px] is the Probability that x_i is 1, given x_(i-5) : i-1
            p_1 = lookup[px][0] / lookup[px][1]

            # For perfect distribution
            # p_1 = 0.5

            p_x = p_1 if x == 1 else (1.0 - p_1)

            # Shannon Entropy Equation
            #H = -(p_0 * math.log2(p_0) + p_1 * math.log2(p_1))

            H = -math.log2(p_x)
            HH += H

            # increment look up tables
            lookup[px][0] += x == 1
            lookup[px][1] += 1

            prevx.append(x)
            prevx = prevx[-NUMBER_OF_BITS:]

    except StopIteration:
        pass

    return (HH/8.0)
    #print("%0.2f bytes of Entropy" % (HH/8.0))

bits = []
perf = []

for i in range(1, 32):
    bits.append(i)
    perf.append(test(i))

perf_dict = {'Lookup Bits': bits, 'Compression Ratio':perf}
perf_df = pd.DataFrame(perf_dict)

gp = ggplot(aes(x='Lookup Bits',y='Compression Ratio'), data = perf_df) + geom_point()+ geom_path()

print(perf_df)
print(gp)
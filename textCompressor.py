# enwik4
# Original Size:    10000
# 7 Zipped:         3719

from collections import defaultdict
import math

enw = open("enwik4", "rb").read()
NUMBER_OF_BITS = 4

def bitgen(x):
    for c in x:
        for i in range(8):
            yield int((c & (0x80>>i)) != 0)

def setgen(x, l):
    bg = bitgen(x)
    ret = []
    while 1:
        ret.append(next(bg))
        ret = ret[-l:]
        if (len(ret) == l):
            yield ret

#https://en.wikipedia.org/wiki/Krichevsky–Trofimov_estimator
lookup = defaultdict(lambda: [0.5,1.0])

sg = setgen(enw, NUMBER_OF_BITS)
bg = bitgen(enw)

# Generating a Probabilistic Suffix Tree
class Node():
    def __init__(self):
        self.c = [0,0]
        self.n = [None, None]

    def getp(self,x):
        t = x[-1]
        if x != [] and self.n[x[-1]] is not None:
            return self.n[x[]].getp(x[:-1])
        return (self.c[t] + 0.5) / (self.c[0] + self.c[1] + 1)
    
    def add(self, x):
        if x == []:
            return
        t = x[-1]
        self.c[t] += 1
        if self.n[t] is None:
            self.n[t] = Node()
        self.n[t].add(x[:-1])

root = Node()

for i in range(15):
    x = next(sg)
    root.getp(x)
    root.add(x)
    print(x)

exit(0)

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

print("%0.2f bytes of Entropy" % (HH/8.0))
enw8 = open("enwik8", "rb").read()

def bitgen(x):
    for c in x:
        for i in range(8):
            yield int((c & (0x80>>i)) != 0)

bg = bitgen(enw8)
for i in range(10):
    print(i, next(bg))
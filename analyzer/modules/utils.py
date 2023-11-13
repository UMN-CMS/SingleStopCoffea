import awkward as ak

def numMatching(a1, a2):
    c = ak.cartesian([a1, a2])
    x, y = ak.unzip(c)
    m = x == y
    return ak.sum(m, axis=1)

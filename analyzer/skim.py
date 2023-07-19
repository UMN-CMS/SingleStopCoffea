from pathlib import Path
import logging
import uproot
import awkward as ak


def isRootCompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak._ext.ArrayType):
        if isinstance(t.type, ak._ext.PrimitiveType):
            return True
        if isinstance(t.type, ak._ext.ListType) and isinstance(
            t.type.type, ak._ext.PrimitiveType
        ):
            return True
    return False


def uprootWriteable(events, fields=None):
    """Restrict to columns that uproot can write compactly"""
    out = {}
    for bname in fields if fields is not None else events.fields:
        print(bname)
        if events[bname].fields:
            print("Making dict")
            d = {
                    n: ak.packed(ak.without_parameters(events[bname][n]))
                    for n in events[bname].fields
                    if isRootCompat(events[bname][n])
                }
            print("Zipping")
            out[bname] = ak.zip(d)
        else:
            out[bname] = ak.packed(ak.without_parameters(events[bname]))
    return out

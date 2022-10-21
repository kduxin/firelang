from argparse import Namespace
from firelang import *

__all__ = ["parse_func", "parse_measure"]


def parse_func(name: str, args: Namespace = Namespace(), **kwargs):
    locals().update(args.__dict__)
    locals().update(kwargs)
    segs = name.split("->")
    funcsegs = []

    for nameseg in segs:
        seg = eval(nameseg)
        funcsegs.append(seg)
    return Sequential(funcsegs)


def parse_measure(name, args: Namespace = Namespace(), **kwargs):
    locals().update(args.__dict__)
    locals().update(kwargs)
    return eval(name)

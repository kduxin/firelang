from __future__ import annotations
from argparse import Namespace
from typing import List, Union, Tuple
from collections import OrderedDict
import os
import json
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, functional as F

from firelang.stack import StackingSlicing
from firelang.measure import DiracMixture, metrics
from firelang.utils.timer import Timer, elapsed
from firelang.utils.optim import Loss
from firelang.utils.log import logger
from firelang.utils.parse import parse_func, parse_measure
from . import FireTensor

# Removed corpusit.Vocab dependency

__all__ = [
    "FireEmbedding",
    "FireWord",
    "FireWordConfig",
    "FIREWord",
]


class FireEmbedding(Module):

    funcs: StackingSlicing
    measures: StackingSlicing
    dim: int

    def __init__(
        self,
        func_template: StackingSlicing,
        measure_template: StackingSlicing,
        vocab_size,
    ):
        Module.__init__(self)
        assert func_template.dim == measure_template.dim
        self.dim = func_template.dim
        self.funcs: StackingSlicing = func_template.restack(vocab_size)
        self.measures: StackingSlicing = measure_template.restack(vocab_size)
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def forward(self, ranks: Tensor = None) -> FireTensor:
        """
        Args:
            ranks (Tensor): (n, ) of word ranks

        Returns:
            (Functional, Measure)
        """
        if ranks is None:
            return FireTensor(
                funcs=self.funcs,
                measures=self.measures,
            )
        else:
            return FireTensor(
                funcs=self.funcs[ranks],
                measures=self.measures[ranks],
            )

    def detect_device(self) -> torch.device:
        return next(iter(self.parameters())).device


class FireWordConfig(dict):
    dim: int
    func: str
    measure: str

    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
        self.__dict__ = self


class FireWord(FireEmbedding):

    config: FireWordConfig
    vocab: object  # Changed from Vocab to generic object

    def __init__(
        self,
        config: FireWordConfig,
        vocab: object,  # Changed from Vocab to generic object
    ):
        FireEmbedding.__init__(
            self,
            func_template=parse_func(config.func, dim=config.dim),
            measure_template=parse_measure(config.measure, dim=config.dim),
            vocab_size=len(vocab),
        )
        self.config = config
        self.i2rank, self.rank2i = self._ranking(vocab)
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def _ranking(self, vocab):
        # Handle both old corpusit.Vocab and new SimpleVocab interface
        if hasattr(vocab, 'i2s_dict'):
            # Old corpusit.Vocab interface
            ids = sorted(vocab.i2s_dict().keys())
        else:
            # New SimpleVocab interface
            ids = sorted(vocab.i2s.keys())
        
        # Convert string IDs to integers if necessary
        if isinstance(ids[0], str):
            ids = [int(id_str) for id_str in ids]
        
        maxid = max(ids)
        i2rank = -np.ones(maxid + 1, dtype=np.int64)
        rank2i = -np.ones(len(ids), dtype=np.int64)
        for rank, idx in enumerate(ids):
            i2rank[idx] = rank
            rank2i[rank] = idx
        return i2rank, rank2i

    def __getitem__(self, words: Union[str, List[str]]) -> FireTensor:
        """
        Args:
            words (str or List[str]): a word or a list of words

        Returns:
            (Functional, Measure)
        """

        s2i = self.vocab.s2i
        # Handle both old and new vocab interfaces
        if hasattr(self.vocab, 'unk_id'):
            unk_id = self.vocab.unk_id
        else:
            # For SimpleVocab, use special_name2i
            unk_id = self.vocab.special_name2i.get('<unk>', 0)
        
        if isinstance(words, str):  # only one word
            ids = [s2i.get(words, unk_id)]
        elif isinstance(words, list):
            ids = [s2i.get(word, unk_id) for word in words]
        else:
            raise TypeError(type(words), words)

        ids = np.array(ids, dtype=np.int64)
        ranks: np.ndarray = self.i2rank[ids]
        ranks = torch.tensor(ranks, device=self.detect_device(), dtype=torch.long)
        return self.forward(ranks)

    def field(
        self, word: str, meshx: Tensor, meshy: Tensor, *mesh_others: Tuple[Tensor]
    ) -> Tensor:
        """Compute the field strength values of the given word, at locations
        specified by `meshx`, `meshy` (in a 2-d field),
        and possibly `mesh_others` (in a higher-dimensional field).

        `meshx` and `meshy` can be, for example, generated with torch.meshgrid.

        Args:
            word (str): the word that generates the field.
            meshx (Tensor): the x-axis coordinates of the locations.
            meshy (Tensor): the y-axis coordinates of the locations.

        Returns:
            Tensor: the field strength values, with the same shape with `meshx`.
        """
        assert len([meshx, meshy, *mesh_others]) == self.dim
        for m in [meshx, meshy, *mesh_others]:
            assert m.shape == meshx.shape
        stack = int(np.prod(meshx.shape))

        ft: FireTensor = self[[word] * stack]
        measure = DiracMixture(self.dim, 1, limits=None, shape=(stack,)).to(
            self.detect_device()
        )
        measure.requires_grad_(False)
        measure.x.copy_(
            torch.cat(
                [x.reshape(-1, 1, 1) for x in [meshx, meshy, *mesh_others]], dim=-1
            )
        )
        measure.m.copy_(torch.ones(stack, 1, dtype=torch.float32))

        outputs = measure.integral(ft.funcs)  # (stack, 1)
        outputs = outputs.view(*meshx.shape)
        return outputs

    @torch.no_grad()
    def most_similar(
        self,
        word: str,
        k: int = 10,
        p: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """Return the most similar `k` words to `word`, as well as the (frequency-adjusted) similarity scores.

        Args:
            word (str): the word of which the most similar words are computed.
            k (int): the number of similar words to return.
            p (float, optional): a exponent controlling the strength of frequency-based adjustment. Defaults to 0.3.

        Returns:
            List[Tuple[str, float]]: the similar words and their frequency-adjusted similar scores.
        """
        w = self[word]
        sims = self.funcs * w.measures + w.funcs * self.measures  # (vocab_size,)

        # adjust with word frequency
        vocab = self.vocab
        if p is not None:
            # Handle both old and new vocab interfaces
            if hasattr(vocab, 'i2count'):
                counts = torch.tensor(
                    [vocab.i2count[self.rank2i[rank]] for rank in range(len(vocab))],
                    dtype=torch.float32,
                    device=sims.device,
                )
            else:
                # For SimpleVocab, we don't have word counts, use uniform weights
                counts = torch.ones(len(vocab), dtype=torch.float32, device=sims.device)
            sims = sims * (counts**p)

        topk = sims.topk(k)
        ranks = topk.indices.data.cpu().numpy()
        values = topk.values.data.cpu().numpy()

        words = [vocab.i2s[self.rank2i[rank]] for rank in ranks]
        return list(zip(words, values))

    def loss_skipgram(
        self, pairs: Tensor, labels: Tensor, args: Namespace = Namespace()
    ) -> Loss:
        """Noise contrastive estimation loss for the SkipGram task.

        Args:
            pairs (Tensor[torch.long]): (n, 2) of word indices.
            labels (Tensor[torch.bool]): (n, ) indicating whether the
                corresponding word pair is a positive or a negative sample.
        """

        x1: FireTensor = self.forward(pairs[..., 0])
        x2: FireTensor = self.forward(pairs[..., 1])

        loss = Loss()

        logits = x1 * x2
        loss_sim = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        loss.add("sim", loss_sim)

        if hasattr(args, "sinkhorn_weight") and args.sinkhorn_weight > 0.0:
            s = metrics.sinkhorn(
                x1.measures,
                x2.measures,
                reg=args.sinkhorn_reg,
                max_iter=args.sinkhorn_max_iter,
                p=args.sinkhorn_p,
                tau=args.sinkhorn_tau,
                stop_threshold=args.sinkhorn_stop_threshold,
            )  # (n,)
            s[~labels] = -s[~labels]
            loss.add("sinkhorn", s * args.sinkhorn_weight)

        if hasattr(args, "l1reg_weight") and args.l1reg_weight > 0.0:
            params1 = [p.reshape(-1) for p in x1.funcs.parameters()]
            params2 = [p.reshape(-1) for p in x2.funcs.parameters()]
            norm = torch.stack(params1 + params2).abs().mean()
            loss.add("l1reg", norm * args.l1reg_weight)

        return loss

    @staticmethod
    def from_pretrained(dirpath, strict: bool = True) -> FireWord:
        dirpath = os.path.abspath(dirpath)
        if not os.path.exists(dirpath):
            raise FileNotFoundError(f"Directory not found at {dirpath}")

        # config
        with open(f"{dirpath}/config.json", "rt") as f:
            config = FireWordConfig(**json.load(f))

        # vocab - load from JSON file
        with open(f"{dirpath}/vocab.json", 'r') as f:
            vocab_data = json.load(f)
        
        # Create SimpleVocab-like object
        class SimpleVocab:
            def __init__(self, s2i, i2s, special_name2i=None):
                self.s2i = s2i
                self.i2s = i2s
                self.special_name2i = special_name2i or {}
            
            def __len__(self):
                return len(self.s2i)
        
        vocab = SimpleVocab(
            s2i=vocab_data['s2i'],
            i2s=vocab_data['i2s'],
            special_name2i=vocab_data.get('special_name2i', {})
        )

        # state_dict
        word = FireWord(config=config, vocab=vocab)
        state_dict = torch.load(f"{dirpath}/pytorch_model.bin")
        word.load_state_dict(state_dict, strict=strict)
        return word

    def save(self, dirpath):
        dirpath = os.path.abspath(dirpath)
        if os.path.exists(dirpath):
            logger.warn(f"Overwriting files in directory {dirpath}.")
        else:
            os.makedirs(dirpath, exist_ok=True)

        # config
        with open(f"{dirpath}/config.json", "wt") as f:
            json.dump(self.config.__dict__, f)

        # vocab - save as JSON
        vocab_data = {
            's2i': self.vocab.s2i,
            'i2s': self.vocab.i2s,
            'special_name2i': getattr(self.vocab, 'special_name2i', {})
        }
        with open(f"{dirpath}/vocab.json", 'w') as f:
            json.dump(vocab_data, f, indent=2)

        # state_dict
        torch.save(self.state_dict(), f"{dirpath}/pytorch_model.bin")


FIREWord = FireWord

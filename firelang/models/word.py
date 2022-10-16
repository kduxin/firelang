from __future__ import annotations
from typing import List, Union, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, functional as F

from firelang import debug_on
from firelang.measure import Measure
from firelang.function import Functional
from firelang.stack import StackingSlicing
from firelang.measure import DiracMixture
from firelang.utils.timer import Timer, elapsed
from firelang.utils.optim import Loss

from corpusit import Vocab

__all__ = [
    "FIREWord",
]


class FIREWord(Module):

    funcs: StackingSlicing
    measures: StackingSlicing
    dim: int
    vocab: Vocab

    def __init__(
        self,
        func_template: StackingSlicing,
        measure_template: StackingSlicing,
        dim,
        vocab: Vocab,
    ):
        super().__init__()
        self.vocab_size = len(vocab)
        self.funcs: StackingSlicing = func_template.restack(self.vocab_size)
        self.measures: StackingSlicing = measure_template.restack(self.vocab_size)

        self.dim = dim
        self.vocab = vocab
        self.i2rank, self.rank2i = self._ranking()

    def _ranking(self):
        ids = sorted(self.vocab.i2s_dict().keys())
        maxid = max(ids)
        i2rank = -np.ones(maxid + 1, dtype=np.int64)
        rank2i = -np.ones(len(ids), dtype=np.int64)
        for rank, idx in enumerate(ids):
            i2rank[idx] = rank
            rank2i[rank] = idx
        return i2rank, rank2i

    def __len__(self):
        return self.vocab_size

    def detect_device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def forward(self, ranks: Tensor) -> Tuple[Functional, Measure]:
        """
        Args:
            ranks (Tensor): (n, ) of word ranks

        Returns:
            (Functional, Measure)
        """
        if not isinstance(ranks, Tensor):
            ranks = torch.tensor(ranks, device=self.detect_device(), dtype=torch.long)

        with Timer(elapsed, "slicing", sync_cuda=debug_on):
            func = self.funcs[ranks]
            measure = self.measures[ranks]
        return FIREWordSlice(func, measure)

    def __getitem__(self, words: Union[str, List[str]]) -> Tuple[Functional, Measure]:
        """
        Args:
            words (str or List[str]): a word or a list of words

        Returns:
            (Functional, Measure)
        """

        s2i = self.vocab.s2i
        unk_id = self.vocab.unk_id
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

        func, _ = self[[word] * stack]
        measure = DiracMixture(self.dim, 1, range=None, stack_size=stack).to(
            self.detect_device()
        )
        measure.requires_grad_(False)
        measure.x.copy_(
            torch.cat(
                [x.reshape(-1, 1, 1) for x in [meshx, meshy, *mesh_others]], dim=-1
            )
        )
        measure.m.copy_(torch.ones(stack, 1, dtype=torch.float32))

        outputs = measure.integral(func)  # (stack, 1)
        outputs = outputs.view(*meshx.shape)
        return outputs

    def loss_skipgram(self, pairs: Tensor, labels: Tensor) -> Loss:
        """Noise contrastive estimation loss for the SkipGram task.

        Args:
            pairs (Tensor[torch.long]): (n, 2) of word indices.
            labels (Tensor[torch.bool]): (n, ) indicating whether the
                corresponding word pair is a positive or a negative sample.
        """

        func1, measure1 = self.forward(pairs[..., 0])
        func2, measure2 = self.forward(pairs[..., 1])

        sim1 = measure2.integral(func1)
        sim2 = measure1.integral(func2)
        logits = sim1 + sim2

        loss = Loss()
        loss_sim = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        loss.add("sim", loss_sim)

        return loss


class FIREWordSlice:
    def __init__(self, funcs: Functional, measures: Measure):
        self.funcs: Functional = funcs
        self.measures: Measure = measures

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return [self.funcs, self.measures][i]

    def __mul__(self, other: FIREWordSlice):
        funcs, measures = self
        funcs_other, measures_other = other
        if id(other) == id(self):
            return measures.integral(funcs, sum=False) * 2
        else:
            return measures_other.integral(funcs, sum=False) + measures.integral(funcs_other, sum=False)

    def __matmul__(self, other: FIREWordSlice):
        funcs, measures = self
        funcs_other, measures_other = other
        if id(other) == id(self):
            mat = measures.integral(funcs, cross=True)
            return mat + mat.T
        else:
            return (
                measures_other.integral(funcs, cross=True)
                + measures.integral(funcs_other, cross=True).T
            )

    def __repr__(self):
        return (
            f"<(func={self.funcs.__class__.__name__}, "
            f"measure={self.measures.__class__.__name__}), "
            f"stack_size={len(self.measures)}>"
        )

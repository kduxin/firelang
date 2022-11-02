from __future__ import annotations
from argparse import Namespace
from typing import List, Union, Tuple, Iterable
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, functional as F

from firelang.function import Functional
from firelang.stack import StackingSlicing
from firelang.measure import DiracMixture
from firelang.utils.timer import Timer, elapsed
from firelang.utils.optim import Loss
from firelang.utils.limits import parse_rect_limits

from corpusit import Vocab

__all__ = [
    "PFIREWord",
]


class PFIREWord(Module):

    funcs: StackingSlicing
    dim: int
    vocab: Vocab

    def __init__(
        self,
        func_template: StackingSlicing,
        limits: float | Tuple[float, float] | List[Tuple[float, float]],
        dim_sizes: int | Tuple[float],
        vocab: Vocab,
    ):
        super().__init__()
        self.vocab_size = len(vocab)
        self.funcs: StackingSlicing = func_template.restack(self.vocab_size)

        self.limits = parse_rect_limits(limits, 2)  # (dim=2, 2)

        if isinstance(dim_sizes, Iterable):
            assert len(dim_sizes) == 2
        else:
            dim_sizes = [dim_sizes] * 2
        self.dim_sizes = dim_sizes

        self.vocab = vocab
        self.i2rank, self.rank2i = self._ranking()

        self.meshx, self.meshy = torch.meshgrid(
            torch.linspace(*self.limits[0].data.cpu().numpy().tolist(), dim_sizes[0]),
            torch.linspace(*self.limits[1].data.cpu().numpy().tolist(), dim_sizes[1]),
        )

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

    def forward(self, ranks: Tensor) -> Functional:
        """
        Args:
            ranks (Tensor): (n, ) of word ranks

        Returns:
            Functional
        """
        if not isinstance(ranks, Tensor):
            ranks = torch.tensor(ranks, device=self.detect_device(), dtype=torch.long)

        with Timer(elapsed, "slicing", sync_cuda=True), Timer(
            elapsed, "slicing", sync_cuda=True, relative=False
        ):
            func = self.funcs[ranks]
        return func

    def __getitem__(self, words: Union[str, List[str]]) -> Functional:
        """
        Args:
            words (str or List[str]): a word or a list of words

        Returns:
            Functional
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

    def measure_on_grid(self, meshx, meshy):
        """
        Returns:
            StackedMeasure: (stack,)
        """
        device = self.detect_device()
        measure = (
            DiracMixture(dim=2, k=np.prod(meshx.shape), range=None, mfix=True)
            .stack(1)
            .to(device)
        )
        measure.requires_grad_(False)
        measure.x.copy_(
            torch.cat(
                [x.reshape(1, -1, 1) for x in [meshx.to(device), meshy.to(device)]],
                dim=-1,
            )
        )
        return measure.to(device)

    def field(self, word):
        stack = np.prod(self.dim_sizes)
        func = self[[word] * stack]
        measure = self.measure_on_grid(stack)
        outputs = measure.integral(func)  # (stack, 1)
        outputs = outputs.view(*self.meshx.shape)
        return outputs.to(self.detect_device())

    def fields_byid(self, ids):
        stack = np.prod(self.dim_sizes)
        func = self.forward(ids)
        measure = self.measure_on_grid(stack)
        outputs = measure.integral(func)  # (stack, 1)
        outputs = outputs.view(*self.meshx.shape)

    def grids(self, words, reshape=False, meshx=None, meshy=None):
        ids = [self.vocab.s2i[w] for w in words]
        ids = torch.tensor(ids, dtype=torch.long, device=self.detect_device())
        return self.grids_byid(ids, reshape=reshape, meshx=meshx, meshy=meshy)

    def grids_byid(self, ids, reshape=False, meshx=None, meshy=None):
        meshx = meshx if meshx is not None else self.meshx
        meshy = meshy if meshy is not None else self.meshy

        measure = self.measure_on_grid(meshx, meshy)
        func = self.forward(ids)
        grid = measure.integral(func, cross=True, sum=False).squeeze(-2)
        if reshape:
            grid = grid.reshape(-1, *meshx.shape)
        return grid

    def loss_skipgram(
        self, pairs: Tensor, labels: Tensor, args: Namespace = Namespace()
    ) -> Loss:
        """Noise contrastive estimation loss for the SkipGram task.

        Args:
            pairs (Tensor[torch.long]): (n, 2) of word indices.
            labels (Tensor[torch.bool]): (n, ) indicating whether the
                corresponding word pair is a positive or a negative sample.
        """
        unique_ids = torch.unique(pairs)
        grids = self.grids_byid(unique_ids)
        # grids: (len(unique_ids), n**2)

        vocabid_to_uniqorder = torch.zeros(
            len(self.vocab), dtype=torch.long, device=pairs.device
        )
        vocabid_to_uniqorder[unique_ids] = torch.arange(
            len(unique_ids), dtype=torch.long, device=pairs.device
        )
        ids = vocabid_to_uniqorder[pairs]  # (stack, 2)
        grid1 = grids[ids[:, 0]]
        grid2 = grids[ids[:, 1]]

        logits = (grid1 * grid2).sum(-1)
        # logits: (stack)

        loss = Loss()
        loss_sim = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        loss.add("sim", loss_sim)
        return loss

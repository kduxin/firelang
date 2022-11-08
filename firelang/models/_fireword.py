from __future__ import annotations
from argparse import Namespace
from typing import List, Union, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, functional as F

from firelang.measure import Measure
from firelang.function import Functional
from firelang.stack import StackingSlicing, IndexLike
from firelang.measure import DiracMixture, metrics
from firelang.utils.timer import Timer, elapsed
from firelang.utils.optim import Loss

from corpusit import Vocab

__all__ = [
    "FIREWord",
    "FIRETensor",
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

    def forward(self, ranks: Tensor) -> FIRETensor:
        """
        Args:
            ranks (Tensor): (n, ) of word ranks

        Returns:
            (Functional, Measure)
        """
        if not isinstance(ranks, Tensor):
            ranks = torch.tensor(ranks, device=self.detect_device(), dtype=torch.long)

        with Timer(elapsed, "slicing", sync_cuda=True), Timer(
            elapsed, "slicing", sync_cuda=True, relative=False
        ):
            func = self.funcs[ranks]
            measure = self.measures[ranks]
        return FIRETensor(func, measure)

    def __getitem__(self, words: Union[str, List[str]]) -> FIRETensor:
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

        outputs = measure.integral(func)  # (stack, 1)
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
            counts = torch.tensor(
                [vocab.i2count[self.rank2i[rank]] for rank in range(len(vocab))],
                dtype=torch.float32,
                device=sims.device,
            )
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

        x1: FIRETensor = self.forward(pairs[..., 0])
        x2: FIRETensor = self.forward(pairs[..., 1])

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

        return loss


class FIRETensor:
    def __init__(self, funcs: Functional, measures: Measure):
        assert funcs.shape == measures.shape
        self.funcs: Functional = funcs
        self.measures: Measure = measures

    def __getitem__(self, index: IndexLike) -> FIRETensor:
        return FIRETensor(self.funcs[index], self.measures[index])

    def view(self, *shape, inplace=False) -> FIRETensor:
        if inplace:
            self.funcs.view(*shape, inplace=True)
            return self
        else:
            return FIRETensor(
                funcs=self.funcs.view(*shape, inplace=False),
                measures=self.measures.view(*shape, inplace=False),
            )

    def __mul__(self, other: FIRETensor):
        if id(other) == id(self):
            return self.measures.integral(self.funcs) * 2
        else:
            return other.measures.integral(self.funcs) + self.measures.integral(
                other.funcs
            )

    def __matmul__(self, other: FIRETensor):
        if id(other) == id(self):
            mat = self.measures.integral(self.funcs, cross=True)
            return mat + torch.transpose(mat, -2, -1)
        else:
            return other.measures.integral(self.funcs, cross=True) + torch.transpose(
                self.measures.integral(other.funcs, cross=True), -2, -1
            )

    def __repr__(self):
        return (
            f"<FIRETensor(funcs={self.funcs.__class__.__name__}, "
            f"measures={self.measures.__class__.__name__}), "
            f"shape={self.funcs.shape}>"
        )

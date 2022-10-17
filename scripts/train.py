from typing import List
import argparse
import os
import sys
import io
import random
import numpy as np
from string import ascii_uppercase, digits
from contextlib import nullcontext
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
from torch.optim import AdamW, Adam, SGD, Adagrad
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from corpusit import Vocab, SkipGramDataset
from firelang import FIREWord
from firelang.utils.parse import parse_func, parse_measure
from firelang.utils.optim import DummyScheduler
from firelang.utils.log import logger
from firelang.utils.timer import elapsed, Timer
from scripts.benchmark import (
    SimilarityBenchmark,
    ALL_WORDSIM_BENCHMARKS,
    load_word_benchmark,
    benchmark_word_similarity,
)


logger.setLevel(level=os.environ.get("LOGLEVEL", "DEBUG").upper())

try:
    import wandb
    from wandb_config import config
    from wandb_config import download_wandb_file
except Exception as e:
    logger.warn(
        "Unable to import wandb for experiment tracking. "
        "Consider executing `pip install wandb`."
    )


_total_elapsed = {}
_total_timer = Timer(_total_elapsed, "total")


@_total_timer
def train(args):
    if args.use_wandb:
        config()
        exp_name = "".join(random.choices(ascii_uppercase + digits, k=8))
        args.savedir = f"{args.savedir}/{exp_name}/"
        wandb.init(name=exp_name, config=args.__dict__)

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    if args.task == "skipgram":
        vocab_args = {
            "min_count": args.min_count,
            "unk": args.unk,
        }
        if args.vocab_path_json is not None:
            vocab = Vocab.from_json(args.vocab_path_bin, **vocab_args)
        elif os.path.exists(args.corpus_path + ".vocab.bin"):
            vocab = Vocab.from_bin(args.corpus_path + ".vocab.bin", **vocab_args)
        elif os.path.exists(args.corpus_path + ".vocab.json"):
            vocab = Vocab.from_json(args.corpus_path + ".vocab.json", **vocab_args)
        else:
            raise ValueError(
                "Vocab not found at `${corpus_path}.vocab.bin` "
                "or at `${corpus_path}.vocab.json`. "
                "You need to build the vocabulary first."
            )
        dataset = SkipGramDataset(
            args.corpus_path,
            vocab=vocab,
            win_size=args.win_size,
            sep=" ",
            mode=args.read_mode,
            subsample=args.subsample,
            power=args.power,
            n_neg=args.n_neg,
        )
    else:
        raise ValueError(f"Failed to recognize task == {args.task}")
    dataloader = dataset.sampler(args.sz_batch, args.seed)
    logger.info(f"Dataset initialized with a dictionary of size {len(vocab)} .")

    set_seed(args.seed)
    if not (args.use_wandb and args.wandb_pretrained):
        func_template = parse_func(args.func, args).to(device)
        measure_template = parse_measure(args.measure, args).to(device)
        if args.model == "FIREWord":
            model = FIREWord(
                func_template,
                measure_template,
                args.dim,
                vocab,
            )
        else:
            raise ValueError(args.model)
    else:
        model = torch.load(
            download_wandb_file(args.wandb_pretrained, "best"),
            map_location="cpu",
        )
        wandb.config.update({"continue": True, "previous_run": args.wandb_pretrained})
    logger.info(model)
    num_parameters = count_parameters(model) // len(vocab)
    logger.info(f"number of parameters for each word: {num_parameters}")
    if args.use_wandb:
        wandb.log({"num_parameters": num_parameters, "vocab_size": len(vocab)})

    model = model.to(device)
    if args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        optimizer = Adagrad(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.n_iters // args.accum_steps + 5,
            div_factor=1.0,
            final_div_factor=20.0,
        )
    else:
        scheduler = DummyScheduler(args.lr)
    logger.info(f"Initialized optimizer and scheduler")
    logger.info(f"  Optimizer: {optimizer}")
    logger.info(f"  Scheduler: {scheduler}")

    benchmarks = ALL_WORDSIM_BENCHMARKS if args.benchmarks == "*" else args.benchmarks
    for bname in benchmarks:
        assert bname in ALL_WORDSIM_BENCHMARKS, f"benchmark {bname} not defined."
    benchmarks: SimilarityBenchmark = {
        bname: load_word_benchmark(bname, lower=args.benchmark_lower)
        for bname in benchmarks
    }

    if args.amp:
        scaler = GradScaler()
        autocaster = autocast()
    else:
        autocaster = nullcontext()
    if args.profile:
        prof = torch.autograd.profiler.profile(use_cuda=True)
    else:
        prof = nullcontext()

    best_iter, best_simscore, best_loss = -1, 0, float("Inf")
    best_savepath = f"{args.savedir}/best"
    for i in range(1, args.n_iters + 1):

        with Timer(elapsed, "prepare", sync_cuda=True):
            inputs, labels = next(dataloader)
            inputs = torch.tensor(
                inputs, dtype=torch.long, device=device
            )  # (sz_batch, 2) int
            labels = torch.tensor(
                labels, dtype=torch.bool, device=device
            )  # labels: (sz_batch, ) bool
        """ ----------------- forward pass -------------------"""
        with prof, autocaster, Timer(elapsed, "forward", sync_cuda=True):
            if args.task == "skipgram":
                if args.model == "FIREWord":
                    model: FIREWord
                    loss = model.loss_skipgram(
                        inputs,
                        labels,
                    )
                else:
                    raise ValueError(args.model)
            else:
                raise ValueError(f"Failed to recognize task == {args.task}")
            total_loss = loss.reduced_total()
            steploss = total_loss / args.accum_steps
        if args.profile:
            logger.debug("----- forward -----")
            logger.debug(prof.key_averages().table(sort_by="self_cpu_time_total"))
        """ ----------------- backward pass -------------------"""
        with prof, Timer(elapsed, "backward", sync_cuda=True):
            if args.amp:
                scaler.scale(steploss).backward()
            else:
                steploss.backward()
            # do not clip, only for acquiring the gradient norm.
            grad_norm = (
                torch.cat([p.grad.data.reshape(-1) for p in model.parameters()])
                .norm()
                .item()
            )

        if args.profile:
            logger.debug("----- backward -----")
            logger.debug(prof.key_averages().table(sort_by="self_cpu_time_total"))
        """ ----------------- optim -------------------"""
        if i % args.accum_steps == 0:
            with Timer(elapsed, "optim/step", sync_cuda=True):
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            with Timer(elapsed, "optim/lrstep", sync_cuda=True):
                scheduler.step()

            with Timer(elapsed, "optim/zerograd", sync_cuda=True):
                model.zero_grad()

        if i % args.eval_interval == 0:

            os.makedirs(args.savedir, exist_ok=True)
            model.eval()

            """--------------- similarity benchmark ---------------"""
            with Timer(elapsed, "benchmark", sync_cuda=True):
                simscores = (
                    benchmark_word_similarity(
                        model,
                        benchmarks,
                    )
                    * 100
                )
            simscore = simscores.mean()
            if simscore > best_simscore:
                best_iter = i
                best_simscore = simscore
                best_loss = total_loss.item()
                torch.save(model, best_savepath)

            if args.task == "skipgram":
                n_pos = labels.sum()
                n_neg = len(labels) - n_pos
            else:
                n_pos = len(labels)
                n_neg = 0
            logger.info(
                f"Iter {i}. Loss={loss}; grad={grad_norm:.3g}; "
                f"lr={scheduler.get_last_lr()[0]:.3g}; "
                f"n={n_pos}+{n_neg}; "
                f"meansim={simscore:.3f}%"
            )
            logger.debug(simscores.to_string(float_format="%5.1f"))
            _print_elapsed(elapsed)
            _total_timer.update()
            _print_elapsed(_total_elapsed)

            if args.use_wandb:
                loginfo = {
                    "iter": i,
                    "eval/loss": total_loss.item(),
                    **{f"eval/loss/{ln}": l.item() for ln, l in loss.reduced_items()},
                    "eval/grad": grad_norm,
                    "eval/simscore": simscore,
                    **{f"eval/simscore/{bn}": score for bn, score in simscores.items()},
                }
                if args.dim == 2:
                    """---------------- visualize ----------------"""
                    if args.model == "FIREWord":
                        fig = visualize_fire(model, args.plot_words)
                    else:
                        raise ValueError(args.model)
                    img = wandb.Image(_fig2array(fig))
                    plt.close(fig)
                    loginfo["wordfig"] = img
                wandb.log(loginfo)
                wandb.save(best_savepath, args.savedir, policy="end")

            model.train()

    model.eval()

    logger.info(
        f"Best iteration: {best_iter}. Similarity score={best_simscore:.3g}, loss={best_loss:.3g}, savepath={best_savepath}"
    )
    if args.use_wandb:
        wandb.log(
            {
                "eval/best_iter": best_iter,
                "eval/best_simscore": best_simscore,
                "eval/best_loss": best_loss,
                "eval/best_savepath": best_savepath,
            }
        )


def _print_elapsed(elapsed, thresh=0.8):
    elapsed = sorted(elapsed.items(), key=lambda x: x[1], reverse=True)
    total = sum([t for name, t in elapsed])
    cum = 0
    lines = []
    for name, t in elapsed:
        line = f"{name:<20} {t:>8.2f}"
        lines.append(line)
        cum += t
        if cum >= total * thresh:
            break
    lines = sorted(lines)
    s = "-- Elapsed --\n" + "\n".join(lines)
    other_time = total - cum
    if other_time >= 1.0:
        s += f'\n{"others":<20} {other_time:>8.2f}'
    logger.debug(s)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _fig2array(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = data.reshape((int(h), int(w), -1))
    return img


@torch.no_grad()
def visualize_fire(model: FIREWord, words: List[str], r: float = 4):
    _, measure = model[words]
    positions = measure.get_x()
    weights = (
        torch.ones(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        if isinstance(measure.m, float)
        else measure.m.abs()
    )
    # positions: (stack_size, n, dim)
    # weights:   (stack_size, n)

    if hasattr(measure, "range") and measure.range is not None:
        r = measure.range
    xmax = max(r, positions[:, :, 0].max().item())
    xmin = min(-r, positions[:, :, 0].min().item())
    ymax = max(r, positions[:, :, 1].max().item())
    ymin = min(-r, positions[:, :, 1].min().item())

    xmesh, ymesh = torch.meshgrid(
        torch.linspace(xmin, xmax, 100), torch.linspace(ymin, ymax, 100)
    )
    score = model.field(words[0], xmesh, ymesh)

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    colors = ["#370665", "#35589A", "#F14A16", "#FC9918"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    xmesh, ymesh, score, positions, weights = list(
        map(lambda x: x.data.cpu().numpy(), [xmesh, ymesh, score, positions, weights])
    )
    cont = ax.contourf(xmesh, ymesh, score)
    handlers = []
    for i in range(len(words)):
        pos, ws, color = positions[i], weights[i], colors[i % len(colors)]
        for (x, y), w in zip(pos, ws):
            (h,) = ax.plot(
                x,
                y,
                "o",
                color=color,
                markersize=_sigmoid(w) * 10,
                markeredgecolor="white",
            )
        handlers.append(h)
    ax.legend(handlers, words)
    fig.colorbar(cont)
    return fig


def parse_arguments():
    parser = argparse.ArgumentParser()

    def boolean_string(s):
        if s not in {"False", "True"}:
            raise ValueError("Not a valid boolean string")
        return s == "True"

    # ----- experiment setting -----
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/corpus/text8",
    )
    parser.add_argument(
        "--vocab_path_json",
        type=str,
        default=None,
    )
    parser.add_argument("--model", type=str, default="FIREWord", choices=["FIREWord"])
    parser.add_argument("--task", type=str, default="skipgram", choices=["skipgram"])

    # ----- fire model settings -----
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Dimension of time series. Only `dim=1` is supported now.",
    )
    parser.add_argument(
        "--func", type=str, default="MLP(args.dim, [8, 2, 8])", help="See parse_func()"
    )
    parser.add_argument("--measure", type=str, default="DiracMixture(args.dim, 1)")
    parser.add_argument(
        "--func_measure",
        type=str,
        default="",
        help="concat of `func` and `measure` with sep=@@. For example: `MLPlanarDiv(dim, 4)@@DiracMixture(dim, 10)`",
    )

    # ----- skipgram parameters -----
    parser.add_argument(
        "--win_size",
        type=int,
        default=10,
        help="Defines a neighborhood [x-win_size, x+win_size)",
    )
    parser.add_argument(
        "--unk",
        type=str,
        default="<unk>",
        help="Should be consistent with the vocabulary.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="All words with fewer counts are regarded as {unk}",
    )
    parser.add_argument(
        "--read_mode",
        type=str,
        default="shuffle",
        choices=["shuffle", "repeat", "onepass"],
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1e-4,
        help="Down-sampling frequent words in positive sampling.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.75,
        help="Up-sampling infrequent words in negative sampling.",
    )
    parser.add_argument(
        "--n_neg",
        type=int,
        default=1,
        help="number of negative samples per positive one",
    )

    # ----- optimize -----
    parser.add_argument("--n_iters", type=int, default=100000)
    parser.add_argument(
        "--sz_batch",
        type=int,
        default=8192,
        help="set it large to put all assets in one batch, or make it small to reduce memory usage",
    )
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr_scheduler", type=str, default="OneCycleLR")
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=10,
        help="Update parameters every several iterations.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="model is evaluated every 20 iterations and the snapshot is saved.",
    )

    # ----- miscellaneous -----
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Use cpu rather than CUDA.")
    parser.add_argument("--savedir", type=str, default="./results/")
    parser.add_argument(
        "--plot_words", type=list, default=["bank", "river", "financial"]
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="*",
        help="List of (word) benchmarks for intermediate evaluation.",
    )
    parser.add_argument(
        "--benchmark_lower",
        type=boolean_string,
        default="True",
        help="convert benchmark texts to lower case.",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Use half precision for accelleration."
    )
    parser.add_argument("--profile", action="store_true", help="CPU/GPU profiling.")

    # ----- wandb -----
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Save all results (and additional visualization results) to wandb.",
    )
    parser.add_argument(
        "--wandb_pretrained",
        type=str,
        default=None,
        help="Load a pre-trained FIRE from wandb, by its ID (e.g., allen/firelang/abcdefghi)",
    )
    parser.add_argument("--tag", type=str, default=None)

    args = parser.parse_args()

    if args.func_measure:
        args.func, args.measure = args.func_measure.split("@@")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    train(args)

from typing import List, Mapping
import os
from collections.abc import Callable
from collections import defaultdict, Counter
from corpusit import Vocab
import nltk
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import multiprocessing
import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score

from firelang.models.word import FIREWord
from firelang.utils.log import logger
from firelang.utils.timer import Timer
from scripts.sentsim import sentsim_as_weighted_wordsim_cuda


__all__ = [
    "SimilarityBenchmark",
    "WordInContextBenchmark",
    "ALL_WORDSIM_BENCHMARKS",
    "ALL_SENTSIM_BENCHMARKS",
    "ALL_WIC_BENCHMARKS",
    "load_word_benchmark",
    "load_all_word_benchmarks",
    "load_sentsim_benchmark",
    "load_all_sentsim_benchmarks",
    "load_wic_benchmark",
    "load_all_wic_benchmarks",
    "benchmark_word_similarity",
    "benchmark_sentence_similarity",
    "benchmark_word_in_context",
]


elapsed = {}


class SimilarityBenchmark:
    def __init__(
        self,
        texts1: List[str],
        texts2: List[str],
        sims: List[float],
        tokenizer: Callable,
    ):
        assert len(texts1) == len(texts2) == len(sims)

        if isinstance(texts1, pd.Series):
            texts1 = texts1.tolist()
        if isinstance(texts2, pd.Series):
            texts2 = texts2.tolist()
        if isinstance(sims, pd.Series):
            sims = sims.tolist()

        self.texts1 = texts1
        self.texts2 = texts2
        self.sims = sims
        self.wordseqs1: List[List[int]] = list(map(tokenizer, texts1))
        self.wordseqs2: List[List[int]] = list(map(tokenizer, texts2))

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        i = self._i
        if i == len(self):
            raise StopIteration
        text1 = self.texts1[i]
        text2 = self.texts2[i]
        wordseq1 = self.wordseqs1[i]
        wordseq2 = self.wordseqs2[i]
        sim = self.sims[i]
        self._i += 1
        return (text1, text2, wordseq1, wordseq2, sim)

    def __len__(self):
        return len(self.texts1)


class WordInContextBenchmark:
    def __init__(
        self,
        texts1: List[str],
        texts2: List[str],
        center_words: List[str],
        labels: List[bool],
        tokenizer: Callable,
    ):
        assert len(texts1) == len(texts2) == len(center_words) == len(labels)

        if isinstance(texts1, pd.Series):
            texts1 = texts1.tolist()
        if isinstance(texts2, pd.Series):
            texts2 = texts2.tolist()
        if isinstance(center_words, pd.Series):
            center_words = center_words.tolist()
        if isinstance(labels, pd.Series):
            labels = labels.tolist()

        self.texts1 = texts1
        self.texts2 = texts2
        self.center_words = center_words
        self.labels = [True if label == "T" else False for label in labels]
        self.wordseqs1: List[List[str]] = list(map(tokenizer, texts1))
        self.wordseqs2: List[List[str]] = list(map(tokenizer, texts2))
        self.center_wordseqs: List[List[int]] = list(map(tokenizer, center_words))

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        i = self._i
        if i == len(self):
            raise StopIteration
        text1 = self.texts1[i]
        text2 = self.texts2[i]
        center_word = self.center_words[i]
        wordseq1 = self.wordseqs1[i]
        wordseq2 = self.wordseqs2[i]
        center_wordseq = self.center_wordseqs[i]
        label = self.labels[i]
        self._i += 1
        return (text1, text2, center_word, wordseq1, wordseq2, center_wordseq, label)

    def __len__(self):
        return len(self.texts1)


""" -------------------- word similarity --------------------"""

DEFAULT_WORDSIM_DIR = "data/word-similarity/monolingual/en"

ALL_WORDSIM_BENCHMARKS = [
    "mc-30",
    "men",
    "mturk-287",
    "mturk-771",
    "rg-65",
    "rw",
    "simlex999",
    "simverb-3500",
    "verb-143",
    "wordsim353-rel",
    "wordsim353-sim",
    "yp-130",
]


def load_word_benchmark(
    name, dirpath=DEFAULT_WORDSIM_DIR, lower=True, tokenizer: Callable = lambda x: [x]
):

    if name == "mc-30":
        dataset = pd.read_csv(f"{dirpath}/mc-30.csv", index_col=0).dropna()
    elif name == "men":
        dataset = pd.read_csv(f"{dirpath}/men.csv", index_col=0).dropna()
        dataset["word1"] = dataset["word1"].apply(lambda x: x.split("-")[0])
        dataset["word2"] = dataset["word2"].apply(lambda x: x.split("-")[0])
    elif name == "mturk-287":
        dataset = pd.read_csv(f"{dirpath}/mturk-287.csv", index_col=0).dropna()
    elif name == "mturk-771":
        dataset = pd.read_csv(f"{dirpath}/mturk-771.csv", index_col=0).dropna()
    elif name == "rg-65":
        dataset = pd.read_csv(f"{dirpath}/rg-65.csv", index_col=0).dropna()
    elif name == "rw":
        dataset = pd.read_csv(f"{dirpath}/rw.csv", index_col=0).dropna()
    elif name == "simlex999":
        dataset = pd.read_csv(f"{dirpath}/simlex999.csv", index_col=0).dropna()
    elif name == "simverb-3500":
        dataset = pd.read_csv(f"{dirpath}/simverb-3500.csv", index_col=0).dropna()
    elif name == "verb-143":
        dataset = pd.read_csv(f"{dirpath}/verb-143.csv", index_col=0).dropna()
    elif name == "wordsim353-rel":
        dataset = pd.read_csv(f"{dirpath}/wordsim353-rel.csv", index_col=0).dropna()
    elif name == "wordsim353-sim":
        dataset = pd.read_csv(f"{dirpath}/wordsim353-sim.csv", index_col=0).dropna()
    elif name == "yp-130":
        dataset = pd.read_csv(f"{dirpath}/yp-130.csv", index_col=0).dropna()
    else:
        return NotImplementedError

    if lower:
        dataset["word1"] = dataset["word1"].str.lower()
        dataset["word2"] = dataset["word2"].str.lower()

    dataset = dataset.dropna()

    return SimilarityBenchmark(
        dataset["word1"].tolist(),
        dataset["word2"].tolist(),
        dataset["similarity"].tolist(),
        tokenizer,
    )


def load_all_word_benchmarks(dirpath=DEFAULT_WORDSIM_DIR, lower=True):
    benchmarks = {}
    for bname in ALL_WORDSIM_BENCHMARKS:
        benchmarks[bname] = load_word_benchmark(bname, dirpath=dirpath, lower=lower)
    return benchmarks


_cache = defaultdict(dict)


@torch.no_grad()
def benchmark_word_similarity(
    model: FIREWord, benchmarks: Mapping[str, SimilarityBenchmark]
):
    vocab: Vocab = model.vocab
    device = model.detect_device()

    scores = {}
    for bname, benchmark in benchmarks.items():
        benchmark: SimilarityBenchmark

        pairs, pairs_str, labels = [], [], []
        oov_words = set()
        for (word1, word2, wseq1, wseq2, sim) in iter(benchmark):
            pairs_str.append((word1, word2))
            labels.append(sim)

            pair = []
            for w in [word1, word2]:
                pair.append(vocab.s2i.get(w, vocab.unk_id))
                if w not in vocab:
                    oov_words.add(w)
            pairs.append(pair)

        labels = np.array(labels)
        pairs = torch.tensor(pairs, device=device, dtype=torch.long)

        if len(oov_words) and not _cache.get(f"oov_reported/{bname}", False):
            logger.warning(
                f"Benchmark {bname}: {len(oov_words)} words "
                f"(in {len(benchmark)} pairs) out of vocabulary\n"
                + ", ".join(list(oov_words))
            )
            _cache[f"oov_reported/{bname}"] = True

        """ similarity """
        with Timer(elapsed, "wordsim/similarity", sync_cuda=True):
            func1, measure1 = model.forward(pairs[..., 0])
            func2, measure2 = model.forward(pairs[..., 1])
            preds = measure1.integral(func2 - func1) + measure2.integral(func1 - func2)

        """ smoothing by standardization """

        def _estimate_mean_var(func, measure):
            sims = (
                measall.integral(func, cross=True)
                + measure.integral(funcall, cross=True).T
            )
            sims = sims - (
                measure.integral(func).reshape(-1, 1)
                + measall.integral(funcall).reshape(1, -1)
            )
            mean, std = sims.mean(dim=1), sims.std(dim=1)
            return mean, std

        with Timer(elapsed, "wordsim/smooth", sync_cuda=True):
            allids = torch.cat([pairs[:, 0], pairs[:, 1]], dim=0)
            funcall, measall = model(allids)

            sims1mean, sims1std = _estimate_mean_var(func1, measure1)
            sims2mean, sims2std = _estimate_mean_var(func2, measure2)

            preds = (preds - sims1mean / 2 - sims2mean / 2) / (
                sims1std * sims2std
            ) ** 0.5
            preds = preds.exp()
            preds = preds.data.cpu().numpy()

        """ spearmann """
        r = spearmanr(labels, preds)
        scores[bname] = r.correlation
    return pd.Series(scores)


""" -------------------- sentence similarity --------------------"""


SENTSIM_BENCHMARKS_PATH = [
    ("STS12-en-test/STS.input.MSRpar.txt", "STS12-en-test/STS.gs.MSRpar.txt"),
    ("STS12-en-test/STS.input.MSRvid.txt", "STS12-en-test/STS.gs.MSRvid.txt"),
    ("STS12-en-test/STS.input.SMTeuroparl.txt", "STS12-en-test/STS.gs.SMTeuroparl.txt"),
    (
        "STS12-en-test/STS.input.surprise.OnWN.txt",
        "STS12-en-test/STS.gs.surprise.OnWN.txt",
    ),
    (
        "STS12-en-test/STS.input.surprise.SMTnews.txt",
        "STS12-en-test/STS.gs.surprise.SMTnews.txt",
    ),
    ("STS13-en-test/STS.input.FNWN.txt", "STS13-en-test/STS.gs.FNWN.txt"),
    ("STS13-en-test/STS.input.headlines.txt", "STS13-en-test/STS.gs.headlines.txt"),
    ("STS13-en-test/STS.input.OnWN.txt", "STS13-en-test/STS.gs.OnWN.txt"),
    # The SMS13-en-test/SMT is not included because it has no `input` file
    ("STS14-en-test/STS.input.deft-forum.txt", "STS14-en-test/STS.gs.deft-forum.txt"),
    ("STS14-en-test/STS.input.deft-news.txt", "STS14-en-test/STS.gs.deft-news.txt"),
    ("STS14-en-test/STS.input.headlines.txt", "STS14-en-test/STS.gs.headlines.txt"),
    ("STS14-en-test/STS.input.images.txt", "STS14-en-test/STS.gs.images.txt"),
    ("STS14-en-test/STS.input.OnWN.txt", "STS14-en-test/STS.gs.OnWN.txt"),
    ("STS14-en-test/STS.input.tweet-news.txt", "STS14-en-test/STS.gs.tweet-news.txt"),
    (
        "STS15-en-test/STS.input.answers-forums.txt",
        "STS15-en-test/STS.gs.answers-forums.txt",
    ),
    (
        "STS15-en-test/STS.input.answers-students.txt",
        "STS15-en-test/STS.gs.answers-students.txt",
    ),
    ("STS15-en-test/STS.input.belief.txt", "STS15-en-test/STS.gs.belief.txt"),
    ("STS15-en-test/STS.input.headlines.txt", "STS15-en-test/STS.gs.headlines.txt"),
    ("STS15-en-test/STS.input.images.txt", "STS15-en-test/STS.gs.images.txt"),
    (
        "STS16-en-test/STS.input.answer-answer.txt",
        "STS16-en-test/STS.gs.answer-answer.txt",
    ),
    ("STS16-en-test/STS.input.headlines.txt", "STS16-en-test/STS.gs.headlines.txt"),
    ("STS16-en-test/STS.input.plagiarism.txt", "STS16-en-test/STS.gs.plagiarism.txt"),
    ("STS16-en-test/STS.input.postediting.txt", "STS16-en-test/STS.gs.postediting.txt"),
    (
        "STS16-en-test/STS.input.question-question.txt",
        "STS16-en-test/STS.gs.question-question.txt",
    ),
]

ALL_SENTSIM_BENCHMARKS = [
    "STS12-en-test/STS.input.MSRpar.txt",
    "STS12-en-test/STS.input.MSRvid.txt",
    "STS12-en-test/STS.input.SMTeuroparl.txt",
    "STS12-en-test/STS.input.surprise.OnWN.txt",
    "STS12-en-test/STS.input.surprise.SMTnews.txt",
    "STS13-en-test/STS.input.FNWN.txt",
    "STS13-en-test/STS.input.headlines.txt",
    "STS13-en-test/STS.input.OnWN.txt",
    "STS14-en-test/STS.input.deft-forum.txt",
    "STS14-en-test/STS.input.deft-news.txt",
    "STS14-en-test/STS.input.headlines.txt",
    "STS14-en-test/STS.input.images.txt",
    "STS14-en-test/STS.input.OnWN.txt",
    "STS14-en-test/STS.input.tweet-news.txt",
    "STS15-en-test/STS.input.answers-forums.txt",
    "STS15-en-test/STS.input.answers-students.txt",
    "STS15-en-test/STS.input.belief.txt",
    "STS15-en-test/STS.input.headlines.txt",
    "STS15-en-test/STS.input.images.txt",
    "STS16-en-test/STS.input.answer-answer.txt",
    "STS16-en-test/STS.input.headlines.txt",
    "STS16-en-test/STS.input.plagiarism.txt",
    "STS16-en-test/STS.input.postediting.txt",
    "STS16-en-test/STS.input.question-question.txt",
]

DEFAULT_SENTSIM_DIR = "data/sent-similarity/STS"


def load_sentsim_benchmark(
    name,
    dirpath=DEFAULT_SENTSIM_DIR,
    tokenizer=nltk.tokenize.word_tokenize,
    lower=True,
):
    if name in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
        tasks = [(x, y) for x, y in SENTSIM_BENCHMARKS_PATH if x.startswith(name)]
    else:
        tasks = [(x, y) for x, y in SENTSIM_BENCHMARKS_PATH if x == name]
    allsentpair, allgs = [], []
    for finput, fgs in tasks:
        lines = open(f"{dirpath}/{finput}", "rt").readlines()
        if lower:
            lines = [line.lower() for line in lines]
        sentpairs = [line.strip().split("\t") for line in lines]
        allsentpair.extend(sentpairs)

        lines = open(f"{dirpath}/{fgs}", "rt").readlines()
        gs = [float(line.strip()) if line.strip() else None for line in lines]
        allgs.extend(gs)

    sents1, sents2 = list(zip(*allsentpair))
    data = pd.DataFrame({"sent1": sents1, "sent2": sents2, "sim": allgs})
    data = data.dropna()
    return SimilarityBenchmark(
        data["sent1"].tolist(),
        data["sent2"].tolist(),
        data["sim"].tolist(),
        tokenizer=tokenizer,
    )


def load_all_sentsim_benchmarks(dirpath=DEFAULT_SENTSIM_DIR, lower=True):
    benchmarks = {}
    for bname in ALL_SENTSIM_BENCHMARKS:
        benchmarks[bname] = load_sentsim_benchmark(bname, dirpath=dirpath, lower=lower)
    return benchmarks


@torch.no_grad()
def benchmark_sentence_similarity(
    model: FIREWord,
    benchmarks: Mapping[str, SimilarityBenchmark],
    sif_alpha=1e-3,
):
    vocab: Vocab = model.vocab

    counts = pd.Series(vocab.counts_dict())
    probs = counts / counts.sum()
    sif_weights: Mapping[str, float] = {
        w: sif_alpha / (sif_alpha + prob) for w, prob in probs.items()
    }

    scores = {}
    for bname, benchmark in benchmarks.items():
        benchmark: SimilarityBenchmark

        sents1 = benchmark.wordseqs1
        sents2 = benchmark.wordseqs2
        allsents = sents1 + sents2
        allsents = [
            [w for w in sent if w in sif_weights and w != vocab.unk]
            for sent in allsents
        ]

        """ similarity """
        with Timer(elapsed, "sentsim/similarity", sync_cuda=True):
            simmat = sentence_simmat(model, allsents, sif_weights)

        """ regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) """
        with Timer(elapsed, "sentsim/regularization"):
            diag = np.diag(simmat)
            simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))

        """ rescaling (smoothing) and exponential """

        def _simmat_rescale(simmat) -> np.ndarray:
            scale = np.abs(simmat).mean(axis=1, keepdims=True)
            simmat = simmat / (scale * scale.T) ** 0.5
            return simmat

        with Timer(elapsed, "sentsim/smooth"):
            simmat = _simmat_rescale(simmat)
            simmat = np.exp(simmat)

        N = len(benchmark)
        preds = [simmat[i, i + N] for i in range(N)]

        """ spearmann """
        r = (
            pd.DataFrame({"target": benchmark.sims, "pred": preds})
            .corr()
            .loc["target", "pred"]
        )
        scores[bname] = r
    return pd.Series(scores)


@torch.no_grad()
def batched_cross_selfsim(model, words, col_batch_size=100):
    _, measures = model[words]
    wordsim = np.zeros((len(words), len(words)), dtype=np.float32)
    for i in range(0, len(words), col_batch_size):
        funcs, _ = model[words[i : i + col_batch_size]]
        _wordsim = measures.integral(funcs, cross=True).data.cpu().numpy()
        wordsim[i : i + col_batch_size, :] += _wordsim
        wordsim[:, i : i + col_batch_size] += _wordsim.T
    return wordsim


def sentence_simmat(model, sents: List[List[str]], sif_weights: Mapping[str, float]):
    allwords = sorted(list(set([w for sent in sents for w in sent])))
    weights = np.array([sif_weights[w] for w in allwords], dtype=np.float32)
    s2i = dict(zip(allwords, range(len(allwords))))

    wordsim = batched_cross_selfsim(model, allwords)

    idseqs = [[s2i[w] for w in sent] for sent in sents]

    sentsim = sentsim_as_weighted_wordsim_cuda(
        wordsim, weights, idseqs, device=model.detect_device().index
    )
    return sentsim


""" -------------------- Word-in-Context classificaiton --------------------"""

ALL_WIC_BENCHMARKS = {
    "train": ("train/train.data.txt", "train/train.gold.txt"),
    # "dev": ("dev/dev.data.txt", "dev/dev.gold.txt"),
    # 'test': ('test/test.data.txt', None),   # gold standard of the test set not available
}

DEFAULT_WIC_DIR = "data/WiC/"


def load_wic_benchmark(
    name, dirpath=DEFAULT_WIC_DIR, lower=True, tokenizer: Callable = nltk.word_tokenize
):

    textpath = os.path.join(dirpath, ALL_WIC_BENCHMARKS[name][0])
    labelpath = os.path.join(dirpath, ALL_WIC_BENCHMARKS[name][1])

    data = pd.read_csv(textpath, header=None, sep="\t")
    data["label"] = pd.read_csv(labelpath, header=None, sep="\t")[0]
    data.columns = ["word", "POS", "index", "text1", "text2", "label"]
    data["word"] = list(map(lambda x: x.lower() if lower else x, data["word"]))
    data["text1"] = list(map(lambda x: x.lower() if lower else x, data["text1"]))
    data["text2"] = list(map(lambda x: x.lower() if lower else x, data["text2"]))
    return WordInContextBenchmark(
        data["text1"], data["text2"], data["word"], data["label"], tokenizer=tokenizer
    )


def load_all_wic_benchmarks(
    dirpath=DEFAULT_WIC_DIR, lower=True, tokenizer: Callable = nltk.word_tokenize
) -> Mapping[str, WordInContextBenchmark]:
    benchmarks = {}
    for bname in ALL_WIC_BENCHMARKS:
        benchmarks[bname] = load_wic_benchmark(
            bname, dirpath=dirpath, lower=lower, tokenizer=tokenizer
        )
    return benchmarks


@torch.no_grad()
def benchmark_word_in_context(
    model, benchmarks: Mapping[str, WordInContextBenchmark], sif_alpha=1e-3
):
    vocab: Vocab = model.vocab

    counts = pd.Series(vocab.counts_dict())
    probs = counts / counts.sum()
    sif_weights: Mapping[str, float] = {
        w: sif_alpha / (sif_alpha + prob) for w, prob in probs.items()
    }

    scores = {}
    for bname, benchmark in benchmarks.items():
        benchmark: WordInContextBenchmark

        sents1 = benchmark.wordseqs1
        sents2 = benchmark.wordseqs2
        N = len(sents1)
        allsents = sents1 + sents2
        allsents = [
            [w for w in sent if w in sif_weights and w != vocab.unk]
            for sent in allsents
        ]

        """ similarity """
        with Timer(elapsed, "wic/similarity", sync_cuda=True):
            simmat = sentence_simmat(model, allsents, sif_weights)

        """ regularization: sim(i,j) <- sim(i,j) - 0.5 * (sim(i,i) + sim(j,j)) """
        with Timer(elapsed, "wic/regularization"):
            diag = np.diag(simmat)
            simmat = simmat - 0.5 * (diag.reshape(-1, 1) + diag.reshape(1, -1))

        """ smoothing by standardization """
        with Timer(elapsed, "wic/smooth"):
            mean1 = np.mean(simmat, axis=1, keepdims=True)
            std1 = np.std(simmat, axis=1, keepdims=True)
            mean0 = np.mean(simmat, axis=0, keepdims=True)
            std0 = np.std(simmat, axis=0, keepdims=True)
            simmat = (simmat - (mean1 + mean0) / 2) / (std0 * std1) ** 0.5

            sentsims = np.array([simmat[i, i + N] for i in range(N)])
            sentsims = np.exp(sentsims)

        threshold = np.median(sentsims)
        preds = sentsims > threshold
        acc = (preds == benchmark.labels).mean()
        scores[bname] = acc
    return pd.Series(scores)


""" -------------------- word polysemy detection --------------------"""


def load_numwordsense(path):
    w2nsense = {}
    with open(path) as f:
        for line in f:
            word, num = line.split()
            num = int(num)
            w2nsense[word] = num
    return w2nsense


def detect_num_clusters_DBSCAN(relwordpos, eps=0.5, minfreq=0.00):
    clu = DBSCAN(eps=eps).fit(relwordpos)
    clusterids = np.array(clu.labels_)
    res = 0
    for w, n in Counter(clusterids).items():
        if w >= 0 and n > len(clusterids) * minfreq:
            res += 1
    return res


def get_relwordpos(allmeasures, model, centerword, k=1000, pca=False):

    func, measure = model[centerword]

    strength = allmeasures.integral(func, cross=True).squeeze()
    potrank = torch.argsort(strength, descending=True)
    relwordids = potrank[:k]

    relwordpos = allmeasures._x[relwordids].squeeze(dim=1)

    relwordpos = relwordpos.data.cpu().numpy()
    if pca:
        relwordpos = PCA(n_components=2).fit_transform(relwordpos)

    return relwordpos


@torch.no_grad()
def benchmark_wordsense_number(
    model: FIREWord,
    w2nsense,
    num_workers=os.cpu_count(),
    eps=0.4,
    nfreqwords=50000,
):
    ctx = multiprocessing.get_context()

    """ get most frequent words """
    vocab = model.vocab
    wordcounts = sorted(vocab.counts_dict().items(), key=lambda x: x[1], reverse=True)
    freqwords = [key for key, val in wordcounts[:nfreqwords]]

    """ polysemy prediction """
    allfuncs, allmeasures = model[freqwords]
    with ctx.Pool(num_workers) as pool:
        labels, blabels, preds, bpreds = [], [], [], []
        async_results = []
        for word in tqdm.tqdm(w2nsense):
            label = w2nsense[word]
            word = word.lower()
            with Timer(elapsed, "get_relwordpos"):
                relwordpos = get_relwordpos(allmeasures, model, word, k=1000, pca=True)
            asy = pool.apply_async(
                detect_num_clusters_DBSCAN,
                kwds={"relwordpos": relwordpos, "eps": eps, "minfreq": 0.000},
            )
            async_results.append(asy)
            labels.append(label)
            blabels.append(True if label >= 2 else False)

        for asy in tqdm.tqdm(async_results):
            pred = asy.get()
            preds.append(pred)
            bpreds.append(True if pred >= 2 else False)

    acc = accuracy_score(blabels, bpreds)
    corr = pd.DataFrame({"pred": preds, "label": labels}).corr().loc["pred", "label"]
    return acc, corr


if __name__ == "__main__":

    device = "cuda"

    for checkpoint in [
        "checkpoints/wacky_mlplanardiv_d2_l4_k1_polysemy",
        "checkpoints/wacky_mlplanardiv_d2_l4_k10",
        "checkpoints/wacky_mlplanardiv_d2_l8_k20",
    ]:
        print(f"=============== Checkpoint `{checkpoint}` ================")
        model = torch.load(checkpoint, map_location=device)

        print("------------- word similarity -------------")
        benchmarks = load_all_word_benchmarks()
        scores = benchmark_word_similarity(model, benchmarks)
        print("mean:", scores.mean())
        print(scores)
        print()

        print("------------- sentence similarity -------------")
        benchmarks = load_all_sentsim_benchmarks()
        scores = benchmark_sentence_similarity(model, benchmarks, sif_alpha=0.001)
        print("mean:", scores.mean())
        print(scores)
        print()

        print("------------- WiC similarity -------------")
        benchmarks = load_all_wic_benchmarks()
        scores = benchmark_word_in_context(model, benchmarks, sif_alpha=0.001)
        print("mean:", scores.mean())
        print(scores)
        print()

        print("elapsed:", elapsed)
        print("\n\n\n")

    checkpoint = "checkpoints/wacky_mlplanardiv_d2_l4_k1_polysemy"
    print(f"========= Word polysemy detection with checkpoint `{checkpoint}` =========")
    w2nsense = load_numwordsense("data/wordnet-542.txt")
    model = torch.load(checkpoint, map_location=device)
    acc, corr = benchmark_wordsense_number(model, w2nsense, eps=0.40)
    print(f"Accuracy = {acc*100:.3g}%, Pearson Correlation Coefficient = {corr:.3g}")
    print()

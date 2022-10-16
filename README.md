# FIRELANG

This repository holds the code for the
[paper](https://www.cl.rcast.u-tokyo.ac.jp/~duxin/files/neurips2022fire.pdf):
- Xin Du and Kumiko Tanaka-Ishii. "Semantic Field of Words Represented as Nonlinear Functions", NeurIPS 2022.

We proposed a new word representation in a functional space rather than a vector
space, called FIeld REpresentation (FIRE). Each word $w$ is represented by a
pair $[\mu, f]$, where $\mu$ is one or multiple locations, represented with a
measure; $f: [-4,4]^2\to \mathbb{R}$ is a nonlinear function implemented by an
individual small neural network for each word.  The similarity between two words
is thus computed by a mutual integral between the words:

$$
\mathrm{sim}(w_1,w_2) = \int f_1~\mathrm{d}\mu_2 + \int f_2~\mathbb{d}\mu_1.
$$

Compared with previous word representation methods, FIRE represents nonlinear
word polysemy while preserving a linear structure for additive semantic compositionality.
The word polysemy is represented by the multimodality of $f$ and by the
(possibly) multiple locations of $\mu$; as for compositionality, it
is represented with functional additivity:

$$ w_1+w_2 = [\mu_1+\mu_2, f_1+f_2]. $$

The similarity between two sentences $\Gamma_1$ and
$\Gamma_2$ is thus computed with the word similarity between the words:

$$ \mathrm{sim}(\Gamma_1, \Gamma_2) = \gamma_1^\mathrm{T} \Sigma \gamma_2, $$

where $\gamma_1$ and $\gamma_2$ are weights assigned to the words in sentence 1 and 2,
respectively; $\Sigma$ is the word similarity matrix: $\Sigma_{ij} = \mathrm{sim}(w_i, w_j)$.

<p align="center">
<img src="assets/img/bank.png" width = "49%"/>
<img src="assets/img/financial+river.png" width = "49%" align=center/>
</p>

Figure (left): words that are frequent and similar to the word `bank`,
visualized in the semantic field of `bank`, when $\mu=m\delta(s)$ is simply a
Dirac's $\delta$ function (scaled by $m$).  The color intensity indicates the
value of the function $f_\mathrm{bank}$ at that location, whereas the circles
indicates the locations $s$ of the words.  The two meanings of `bank` are
naturally separated with FIRE.  Figure (right): overlapped semantic fields of
`river` and `financial`, and their locations $s$. The shape resembles that of
`bank` in the image above, indicating FIRE's property of compositionality.


# Environment preparation

- Python >= 3.7
- Packages
  ```bash
  # With CUDA 11 
  $ pip install -r requirements-cu11.txt

  # With CUDA 10
  $ pip install -r requirements-cu10.txt
  ```

- If you are using Windows or MacOS, you need to install the Rust compiling toolchain
  ([cargo](https://www.rust-lang.org/tools/install)) in advance,
  which is used to compile the `corpusit` package from source.


# Evaluating pre-trained FIRE word representations

## 1. Download pre-trained models
We provide the following pre-trained FIRE models:
- [$D=2,L=4,L=1$](https://www.cl.rcast.u-tokyo.ac.jp/~duxin/firelang/pretrained/word/wacky_mlplanardiv_d2_l4_k1_polysemy.gz) (23 parameters per word)
- [$D=2,L=4,L=10$](https://www.cl.rcast.u-tokyo.ac.jp/~duxin/firelang/pretrained/word/wacky_mlplanardiv_d2_l4_k10.gz) (50 parameters per word)
- [$D=2,L=8,L=20$](https://www.cl.rcast.u-tokyo.ac.jp/~duxin/firelang/pretrained/word/wacky_mlplanardiv_d2_l8_k20.gz) (100 parameters per word)

You can run the following to download the three.
```bash
$ bash scripts/benchmark/1_download_pretrained.sh
```
The models will be downloaded to `checkpoints/` and decompressed.

The saved models can be reloaded by
```python
import torch
model = torch.load('checkpoints/wacky_mlplanardiv_d2_l4_k10')
```

## 2. Run benchmarks

Execute the benchmarking script as follows:
```bash
$ bash scripts/benchmark/2_run_benchmark.sh
```



# Training FIRE

## Configuring WanDB accounts
We integrated [WanDB](https://wandb.ai) functionalities in the training program
for experiment management and visualization.
So by default, you need to do the following three steps to enable those functionalities:
- Create `wandb_config.py` from the template `wandb_config.template.py`
- [Register an WanDB account](https://wandb.ai)
- Fill in your username (`WANDB_ENTITY`) and token (`WANDB_API_KEY`) in `wandb_config.py`

If you do not plan to use WanDB, you will have to delete the argument *--use_wandb*
in `scripts/text8/4_train.sh` and `scripts/wacky/4_train.sh`

## Training FIRE on Text8
We provide scripts in `/scripts/text8/` to train a FIRE model on the *text8* dataset.
Text8 is smaller (~100MB) and is publicly available.
```bash
# download the text8 corpus
$ bash scripts/text8/1_download_text8.sh

# tokenize the corpus with the NLTK tokenizer
$ bash scripts/text8/2_tokenize.sh

# build a vocabulary with the tokenized corpus
$ bash scripts/text8/3_build_vocab.sh

# training from scratch
$ bash scripts/text8/4_train.sh
# This would takes 2-3 hours, so it is recommended to run the process in the background.
# For example:
# $ CUDA_VISIBLE_DEVICES=0 nohup bash scripts/text8/4_train.sh > log.train.wacky.log 2>&1 &
```

The training process is carried out with the SkipGram method.
For fast sampling from the tokenized corpus in the SkipGram way, we used
another python package [`corpusit`](https://github.com/kduxin/corpusit)
that is written in Rust (and binded with [PyO3](https://github.com/PyO3/pyo3)).
On Windows or MacOS, installing `corpusit` with `pip` will compile the
package from source code; in this case, you need to have
[cargo](https://www.rust-lang.org/tools/install) installed in advance.

## WaCKy
The *WaCKy* corpus is a concatenation of two corpora `ukWaC` and `WaCkypedia_EN`.
Both are provided at https://wacky.sslmit.unibo.it/doku.php?id=download via request.

After you get the two corpora, put the concatenated file at
`/data/corpus/wacky/wacky.txt`.  Then, you can run scripts under `/scripts/wacky/`
(for tokenization, vocabulary construction, and training) to start training a
FIRE on the concatenated corpus. The process takes 10-20 hours depending on the hardware.



# Parallelized neural networks
A challenge for implementing FIRE is to parallize
the evaluation of (neural-network) functions $f$, especially when each function has its own input.

The usual way of using a neural network NN is to process a data batch at a time,
that is the parallelization of $\text{NN}(x_1)$, $\text{NN}(x_2)$...  Recent
advances in deep learning packages provide a new paradigm to parallelize the
computation of $\text{NN}_1(x)$, $\text{NN}_2(x)$...  such as the
[vmap](https://pytorch.org/tutorials/prototype/vmap_recipe.html) method in JAX
and PyTorch.

In FIRE-based language models, we instead require the parallelization of both
neural networks and data.  The desired behaviors should include:
- paired mode: output a vector. 
  - $\text{NN}_1(x_1)$, $\text{NN}_2(x_2)$, $\text{NN}_3(x_3)$...

  In analog to element-wise multiplication $\text{NN}*x$,
  where $\text{NN}=[\text{NN}_1,\text{NN}_2,\cdots,]^\text{T}$,
  and $x=[x_1,x_2,\cdots]^\text{T}$
- cross mode: output a matrix. 
  - $\text{NN}_1(x_1)$, $\text{NN}_1(x_2)$, $\text{NN}_1(x_3)$...
  - $\text{NN}_2(x_1)$, $\text{NN}_2(x_2)$, $\text{NN}_2(x_3)$...
  - $\text{NN}_3(x_1)$, $\text{NN}_3(x_2)$, $\text{NN}_3(x_3)$...
  - $\cdots$

  In analog to matrix multiplication of vectors: $\text{NN}$ @ $x$.

We call $\text{NN}=[\text{NN}_1,\text{NN}_2,\cdots,\text{NN}_n]$ a stacked
function.

To store the parameters for all words, $n$ is equal to the size
of vocabulary. Each time when we want the representation for a subset
of all words, `slicing` is required to extract the parameters for these
words and recombine them into a new stacked function.

## Words as vectors

For word-vector representations, `slicing` is natively supported
for the matrix $V\in\mathbb{R}^{N\times D}$ whose rows are word vectors.
The computation of the paired similarity is a batched inner product,
and that of the cross similarity is simply a matrix multiplication.
For example:
- Slicing:
  ```python
  vecs1 = V[[id_apple, id_pear, ...]]      # (n1, D)
  vecs2 = V[[id_iphone, id_fruit, ...]]    # (n2, D)
  ```
- Computation of paired similarity (where n1 == n2 must hold):
  ```python
  sim = (vecs1 * vecs2).sum(-1)            # (n1,)
  ```
- Computation of cross similarity (n1 and n2 can be different):
  ```python
  sim = vecs1 @ vecs2.T                    # (n1, n2)
  ```

## FIRE: neural network as a "vector"

In this repository, we provide an analogous implementation for parallelizing neural networks.
In FIRE, each neural network (a function or a measure) is treated like a vector.
Multiple neural networks are `stacked` like the stacking of vectors into a matrix.

In FIRE, the `slicing` and similarity computation are done in a similar way to vectoral.

- Slicing:
  ```python
  funcs1, measures1 = model[["apple", "pear", ...]]       # (n1, D)
  funcs2, measures2 = model[["iphone", "fruit", ...]]     # (n1, D)
  ```
- Computation of paired similarity (where n1 == n2):
  ```python
  sim = measures2.integral(funcs1)    # (n1,)
      + measures1.integral(funcs2)    # (n2,)
  ```
- Computation of cross similarity:
  ```python
  sim = measures2.integral(funcs1, cross=True)    # (n1, n2)
      + measures1.integral(funcs2, cross=True).T  # (n2, n1) -> transpose -> (n1, n2)
  ```

In addition to the way above where `integral` must be explicitly invoked,
a more friendly way is also provided, as below:
```python
# paired similarity
# sim: (n1,)
sim = (funcs1 * measures2)       # (n1, K)    where K is the number of locations in the measure
      .sum(-1)
    + (measures1 * funcs2)       # (n2, K)
      .sum(-1)

# cross similarity
sim = funcs1 @ measures2 + measures1 @ funcs2     # (n1, n2)
```

Furthermore, the two steps above can be done in one line:
```python
sim = model[["apple", "pear", "melon"]] @ model[["iphone", "fruit"]]     # (3, 2)

# which is equivalent to below:
# slice1 = model[["apple", "pear", "melon"]]
# slice2 = model[["iphone", "fruit"]]
# funcs1, measures1 = slice1
# funcs2, measures2 = slice2
# sim = measures2.integral(funcs1, cross=True) + measures1.integral(funcs2, cross=True).T
```


## Combination of functions via arithmetics
For the functions in a FIRE, we implemented arithmetic operators to make the
(stacked) functions look more like a vector.

For example, the regularization of the similarity scores by the following formula

$$ \text{sim}(w_i,w_j) \leftarrow \text{sim}(w_i,w_j) - \int f_i ~\mathrm{d}\mu_i - \int f_j ~\mathrm{d}\mu_j, $$

is done by:

```python
funcs1, measures1 = [["apple", "pear"]]
funcs2, measures2 = [["fruit", "iphone"]]
sim_reg = measures2.integral(funcs1) + measures1.integral(funcs2) \
        - measures1.integral(funcs1) - measures2.integral(funcs2)
```
or equivalently:
```python
sim_reg = ((funcs2 - funcs1) * measures1 + (funcs1 - funcs2) * measures2).sum(-1)
```
where `funcs2 - funcs1` produces a new functional.



# Citation
Please cite the following paper:
```bibtex
@inproceedings{
  du2022semantic,
  title={Semantic Field of Words Represented as Non-Linear Functions},
  author={Xin Du and Kumiko Tanaka-Ishii},
  booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022},
}
```
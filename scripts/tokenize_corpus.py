import os

def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="data/corpus/text8/text8")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cased", action="store_true")
    parser.add_argument("--sos", type=str, default="<s>")
    parser.add_argument("--eos", type=str, default="</s>")
    parser.add_argument("--linesep", type=str, default="\n")
    parser.add_argument("--tokensep", type=str, default=" ")

    return parser.parse_args()


def run(args):
    from nltk import word_tokenize
    import tqdm
    from scripts.multiproc_yield import yielder
    import logging
    import json

    logger = logging.getLogger()
    path = os.path.abspath(args.raw_path)
    dirpath = os.path.dirname(path)
    filename = os.path.basename(path)
    savepath = (
        f"{dirpath}/{filename}" + ("" if args.cased else ".uncased") + ".tokens"
    )
    logger.info(f"Tokenized corpus to be saved at {savepath}")

    argsavepath = savepath + '.args'
    with open(argsavepath, "wt") as f:
        json.dump(args.__dict__, f, indent=2)
    logger.info(f"Arguments saved at {argsavepath}")

    def run_tokenize(corpus_path, save_path, args, max_size_bytes=1024 * 1024 * 1024):
        logger.info("Tokenizing...")

        def _tokenize(line, sos, eos):
            if not args.cased:
                line = line.lower()
            line = line.strip()
            if not line:
                return []
            else:
                return [sos] + word_tokenize(line) + [eos]

        def linecutter(lines, maxlen=10000):
            for line in lines:
                nseg = (len(line) + maxlen - 1) // maxlen
                for i in range(nseg):
                    yield line[i * maxlen : (i + 1) * maxlen]

        ntokens = 0
        fout = open(save_path, "wt")
        with open(corpus_path, "rt") as f:
            for tokens in tqdm.tqdm(
                yielder(
                    linecutter(f),
                    _tokenize,
                    num_workers=args.num_workers,
                    additional_kwds={"sos": args.sos, "eos": args.eos},
                    max_size_bytes=max_size_bytes,
                )
            ):
                if tokens:
                    ntokens += len(tokens)
                    fout.write(args.tokensep.join(tokens) + args.linesep)
        logger.info(f"{ntokens} tokens in saved.")

        fout.close()

    run_tokenize(path, savepath, args)
    logger.info("Finished.")


if __name__ == "__main__":
    args = parse_args()
    run(args)

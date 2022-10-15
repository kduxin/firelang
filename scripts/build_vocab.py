import argparse
import corpusit

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_corpus', type=str)
parser.add_argument('--min_count', type=int, default=5)
parser.add_argument('--infreq_replace', type=str, default="<unk>")
args = parser.parse_args()

vocab = corpusit.Vocab.build(args.path_to_corpus, min_count=args.min_count, unk=args.infreq_replace)
print('vocab size:', len(vocab))
print(vocab)
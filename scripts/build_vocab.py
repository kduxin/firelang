import argparse
import json
from collections import Counter

def build_vocab_and_counts(corpus_path, min_count=5, unk="<unk>"):
    """Build vocabulary and word counts from corpus file."""
    print(f"Building vocabulary from {corpus_path}...")
    
    # Read corpus and count words
    word_counts = Counter()
    total_words = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                if word:  # Skip empty strings
                    word_counts[word] += 1
                    total_words += 1
    
    print(f"Total words processed: {total_words}")
    print(f"Unique words before filtering: {len(word_counts)}")
    
    # Filter by min_count
    filtered_words = {word: count for word, count in word_counts.items() if count >= min_count}
    print(f"Words after min_count={min_count} filtering: {len(filtered_words)}")
    
    # Create vocabulary
    words = sorted(filtered_words.keys())
    s2i = {word: i for i, word in enumerate(words)}
    i2s = {i: word for i, word in enumerate(words)}
    
    # Add special tokens
    special_name2i = {unk: len(words)}
    s2i[unk] = len(words)
    i2s[len(words)] = unk
    
    vocab_data = {
        's2i': s2i,
        'i2s': i2s,
        'special_name2i': special_name2i,
        'total_words': total_words
    }
    
    # Create word counts with integer IDs
    word_counts_with_ids = {s2i[word]: count for word, count in filtered_words.items()}
    
    return vocab_data, word_counts_with_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_corpus', type=str, required=True)
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--infreq_replace', type=str, default="<unk>")
    args = parser.parse_args()
    
    # Build vocabulary and word counts
    vocab_data, word_counts = build_vocab_and_counts(
        args.path_to_corpus, 
        min_count=args.min_count, 
        unk=args.infreq_replace
    )
    
    # Save vocabulary
    vocab_path = args.path_to_corpus + ".vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")
    
    # Save word counts
    word_counts_path = args.path_to_corpus + ".word_counts.json"
    with open(word_counts_path, 'w') as f:
        json.dump(word_counts, f, indent=2)
    print(f"Saved word counts to {word_counts_path}")
    
    print(f'Vocab size: {len(vocab_data["s2i"])}')
    print(f'Word counts: {len(word_counts)}')

if __name__ == "__main__":
    main()
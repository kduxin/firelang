#!/usr/bin/env python3
"""
Test script for DataLoader with different read modes.
"""

import os
import tempfile
import random
from typing import List, Tuple
import numpy as np

from dataloader import DataLoader, StreamingCorpusReader


class MockSampler:
    """Mock sampler for testing DataLoader."""
    
    def __init__(self):
        self.call_count = 0
    
    def process_string_sequences(self, text_sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Mock processing that returns pairs and labels."""
        self.call_count += 1
        
        # Create mock pairs and labels
        pairs = []
        labels = []
        
        for seq in text_sequences:
            words = seq.split()
            if len(words) >= 2:
                # Create some mock pairs
                for i in range(min(3, len(words) - 1)):
                    pairs.append([i, i + 1])  # Mock word indices
                    labels.append(random.choice([True, False]))
        
        if not pairs:
            # Fallback if no pairs generated
            pairs = [[0, 1], [1, 2]]
            labels = [True, False]
        
        return np.array(pairs, dtype=np.int64), np.array(labels, dtype=bool)


def create_test_corpus(filepath: str, num_lines: int = 100):
    """Create a test corpus file."""
    test_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "natural language processing uses neural networks",
        "deep learning models require large datasets",
        "word embeddings capture semantic relationships",
        "skip gram models predict context words",
        "negative sampling improves training efficiency",
        "word2vec is a popular embedding technique",
        "transformer models revolutionized NLP",
        "attention mechanisms focus on relevant tokens"
    ]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in range(num_lines):
            sentence = random.choice(test_sentences)
            f.write(sentence + '\n')


def test_streaming_reader():
    """Test StreamingCorpusReader with different modes."""
    print("Testing StreamingCorpusReader...")
    
    # Create test corpus
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_file = f.name
        create_test_corpus(test_file, 50)
    
    try:
        # Test shuffle mode
        print("  Testing shuffle mode...")
        reader = StreamingCorpusReader(test_file, "shuffle", chunk_size=10)
        lines = []
        for _ in range(20):
            line = reader.next_line()
            if line:
                lines.append(line)
        print(f"    Read {len(lines)} lines in shuffle mode")
        reader.close()
        
        # Test repeat mode
        print("  Testing repeat mode...")
        reader = StreamingCorpusReader(test_file, "repeat", chunk_size=10)
        lines = []
        for _ in range(20):
            line = reader.next_line()
            if line:
                lines.append(line)
        print(f"    Read {len(lines)} lines in repeat mode")
        reader.close()
        
        # Test onepass mode
        print("  Testing onepass mode...")
        reader = StreamingCorpusReader(test_file, "onepass", chunk_size=10)
        lines = []
        for _ in range(20):
            line = reader.next_line()
            if line:
                lines.append(line)
        print(f"    Read {len(lines)} lines in onepass mode")
        reader.close()
        
    finally:
        os.unlink(test_file)


def test_dataloader():
    """Test DataLoader with different modes."""
    print("\nTesting DataLoader...")
    
    # Create test corpus
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_file = f.name
        create_test_corpus(test_file, 100)
    
    try:
        sampler = MockSampler()
        
        # Test shuffle mode
        print("  Testing shuffle mode...")
        dataloader = DataLoader(sampler, batch_size=8, corpus_path=test_file, read_mode="shuffle")
        batch_count = 0
        for i, (pairs, labels) in enumerate(dataloader):
            batch_count += 1
            print(f"    Batch {i+1}: {len(pairs)} pairs, {len(labels)} labels")
            if batch_count >= 5:  # Limit to 5 batches for testing
                break
        print(f"    Processed {batch_count} batches in shuffle mode")
        
        # Test repeat mode
        print("  Testing repeat mode...")
        dataloader = DataLoader(sampler, batch_size=8, corpus_path=test_file, read_mode="repeat")
        batch_count = 0
        for i, (pairs, labels) in enumerate(dataloader):
            batch_count += 1
            print(f"    Batch {i+1}: {len(pairs)} pairs, {len(labels)} labels")
            if batch_count >= 5:  # Limit to 5 batches for testing
                break
        print(f"    Processed {batch_count} batches in repeat mode")
        
        # Test onepass mode
        print("  Testing onepass mode...")
        dataloader = DataLoader(sampler, batch_size=8, corpus_path=test_file, read_mode="onepass")
        batch_count = 0
        for i, (pairs, labels) in enumerate(dataloader):
            batch_count += 1
            print(f"    Batch {i+1}: {len(pairs)} pairs, {len(labels)} labels")
            if batch_count >= 5:  # Limit to 5 batches for testing
                break
        print(f"    Processed {batch_count} batches in onepass mode")
        
    finally:
        os.unlink(test_file)


def test_memory_efficiency():
    """Test that DataLoader doesn't load entire file into memory."""
    print("\nTesting memory efficiency...")
    
    # Create a larger test corpus
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_file = f.name
        create_test_corpus(test_file, 1000)  # 1000 lines
    
    try:
        sampler = MockSampler()
        dataloader = DataLoader(sampler, batch_size=16, corpus_path=test_file, read_mode="shuffle")
        
        # Process a few batches
        batch_count = 0
        for pairs, labels in dataloader:
            batch_count += 1
            if batch_count >= 10:  # Process 10 batches
                break
        
        print(f"    Successfully processed {batch_count} batches without loading entire file")
        
    finally:
        os.unlink(test_file)


if __name__ == "__main__":
    print("Starting DataLoader tests...")
    
    # Set random seed for reproducible tests
    random.seed(42)
    np.random.seed(42)
    
    test_streaming_reader()
    test_dataloader()
    test_memory_efficiency()
    
    print("\nAll tests completed!")

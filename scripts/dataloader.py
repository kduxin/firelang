from __future__ import annotations
from typing import List, Optional, Tuple
from collections import deque
import random
import numpy as np


class StreamingCorpusReader:
    """Streaming corpus reader that supports different reading modes."""
    
    def __init__(self, corpus_path: str, mode: str, chunk_size: int):
        supported_modes = {"shuffle", "repeat", "onepass"}
        if mode not in supported_modes:
            raise ValueError(f"Unsupported read mode: {mode}")
        self.corpus_path = corpus_path
        self.mode = mode
        self.chunk_size = max(1, chunk_size)
        self.file = open(self.corpus_path, "r", encoding="utf-8")
        self.buffer = deque()
        self.exhausted = False
        if mode == "shuffle":
            self._fill_buffer()

    def _fill_buffer(self):
        """Fill buffer with lines from file."""
        if self.exhausted:
            return
        lines = []
        while len(lines) < self.chunk_size:
            line = self.file.readline()
            if not line:
                if not lines:
                    if self.mode == "onepass":
                        self.exhausted = True
                        self.file.close()
                        break
                    self.file.seek(0)
                    line = self.file.readline()
                    if not line:
                        self.exhausted = True
                        self.file.close()
                        break
                    continue
                if self.mode == "onepass":
                    self.exhausted = True
                    self.file.close()
                else:
                    self.file.seek(0)
                break
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
        if not lines:
            return
        if self.mode == "shuffle":
            random.shuffle(lines)
        self.buffer.extend(lines)

    def next_line(self) -> Optional[str]:
        """Get next line from corpus."""
        if self.mode == "shuffle":
            while not self.buffer and not self.exhausted:
                self._fill_buffer()
            if not self.buffer:
                return None
            return self.buffer.popleft()

        while True:
            if self.exhausted:
                return None
            line = self.file.readline()
            if not line:
                if self.mode == "onepass":
                    self.exhausted = True
                    self.file.close()
                    return None
                self.file.seek(0)
                line = self.file.readline()
                if not line:
                    self.exhausted = True
                    self.file.close()
                    return None
            stripped = line.strip()
            if stripped:
                return stripped

    def close(self):
        """Close the file handle."""
        if not self.file.closed:
            self.file.close()


class DataLoader:
    """DataLoader for streaming corpus reading with different modes."""
    
    def __init__(self, sampler, batch_size: int, corpus_path: str, read_mode: str):
        self.sampler = sampler
        self.batch_size = batch_size
        self.read_mode = read_mode
        self.current_pairs = []
        self.current_labels = []
        self.current_pos = 0
        chunk_size = max(1024, batch_size)
        self.reader = StreamingCorpusReader(
            corpus_path=corpus_path,
            mode=read_mode,
            chunk_size=chunk_size,
        )
        self._fallback_sequence = "the of and to in is"

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "reader"):
            self.reader.close()

    def _fetch_sequences(self, target_count: int) -> List[str]:
        """Fetch sequences from corpus."""
        sequences = []
        while len(sequences) < target_count:
            line = self.reader.next_line()
            if line is None:
                break
            sequences.append(line)
        if not sequences:
            if self.read_mode in {"repeat", "shuffle"}:
                sequences = [self._fallback_sequence] * max(1, target_count)
        return sequences

    def _load_more_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load more pairs from corpus."""
        text_sequences = self._fetch_sequences(self.batch_size)
        if not text_sequences:
            return (
                np.array([], dtype=np.int64).reshape(0, 2),
                np.array([], dtype=bool),
            )

        pairs, labels = self.sampler.process_string_sequences(text_sequences)

        if len(pairs) == 0 or len(labels) == 0:
            print("Warning: Empty batch from process_string_sequences")
            return (
                np.array([], dtype=np.int64).reshape(0, 2),
                np.array([], dtype=bool),
            )

        return pairs, labels
    
    def __iter__(self):
        """Make DataLoader iterable."""
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch of data."""
        # Check if we need to load more pairs
        if self.current_pos >= len(self.current_pairs):
            # Load new pairs
            new_pairs, new_labels = self._load_more_pairs()
            if len(new_pairs) == 0:
                # If no new pairs, try again
                new_pairs, new_labels = self._load_more_pairs()
                if len(new_pairs) == 0:
                    raise StopIteration
            
            self.current_pairs = new_pairs
            self.current_labels = new_labels
            self.current_pos = 0
        
        # Return a batch of pairs
        start_idx = self.current_pos
        end_idx = min(start_idx + self.batch_size, len(self.current_pairs))
        
        batch_pairs = self.current_pairs[start_idx:end_idx]
        batch_labels = self.current_labels[start_idx:end_idx]
        
        self.current_pos = end_idx
        
        return batch_pairs, batch_labels

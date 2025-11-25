#!/usr/bin/env python3
"""
C99 Text Segmentation Algorithm for Generic Conversation Transcripts

This module implements the C99 algorithm for topic segmentation, which can be used
to automatically segment transcripts based on semantic coherence.

Original Algorithm from https://github.com/logological/C99
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, List, Tuple
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import re
from collections import Counter

logger = logging.getLogger(__name__)

class C99:
    """
    C99 segmentation algorithm
    Reference:
        "Advances in domain independent linear text segmentation"
    """
    def __init__(self, window: int = 6, std_coeff: float = 1.2):
        """
        window: rank window parameter. The effective mask size is (2*window - 1),
                so window=6 gives an 11x11 rank mask as in Choi (2000).
        std_coeff: 'c' in the paper's threshold μ + c * sqrt(ν). Default 1.2 to
                   match the original recommendation.
        """
        self.window = window
        self.sim: Optional[npt.NDArray[np.float64]] = None
        self.rank: Optional[npt.NDArray[np.float64]] = None
        self.sm:Optional[npt.NDArray[np.float64]] = None
        self.std_coeff = std_coeff

    def segment(self, document_embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """
        Segment document using C99 algorithm
        
        Args:
            document_embeddings: array with embeddings of document sentences (or other segmentation unit)
            
        Returns:
            Array of zeros and ones, i-th element denotes whether exists a boundary right before paragraph i(0 indexed)
            
        Raises:
            ValueError: If input is invalid
        """
        # Input validation
        if document_embeddings.size == 0:
            return np.array([], dtype=np.int64)
        
        if len(document_embeddings.shape) != 2:
            raise ValueError("document_embeddings must be 2D array")
        
        if np.any(np.isnan(document_embeddings)) or np.any(np.isinf(document_embeddings)):
            raise ValueError("document_embeddings contains NaN or infinite values")
        
        # do not segment documents with less than 3 embeddings
        if document_embeddings.shape[0] <= 3:
            out = np.zeros(document_embeddings.shape[0], dtype=np.int64)
            if document_embeddings.shape[0] > 0:
                out[0] = 1
            return out

        # step 1, preprocessing
        n = document_embeddings.shape[0]

        # step 2, compute similarity matrix with numerical stability
        self.sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                norm_i = np.linalg.norm(document_embeddings[i])
                norm_j = np.linalg.norm(document_embeddings[j])
                norm = norm_i * norm_j
                
                if norm < 1e-12:  # Handle zero/near-zero vectors
                    cosine_similarity = 0.0
                else:
                    cosine_similarity = np.dot(document_embeddings[i], document_embeddings[j]) / norm
                    # Clamp to valid range to handle numerical errors
                    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
                
                self.sim[i][j] = self.sim[j][i] = cosine_similarity

        # step 3, compute rank matrix & sum matrix
        # Effective half-width; window defines (2*window - 1) mask size.
        # window=6 -> half=5 => 11x11 mask in the middle
        half = self.window - 1
        
        self.rank = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                r1 = max(0, i - half)
                r2 = min(n - 1, i + half)
                c1 = max(0, j - half)
                c2 = min(n - 1, j + half)
                
                sublist = self.sim[r1:(r2 + 1), c1:(c2 + 1)].flatten()
                lowlist = [x for x in sublist if x < self.sim[i][j]]
                
                num_elems = (r2 - r1 + 1) * (c2 - c1 + 1)
                self.rank[i][j] = len(lowlist) / float(num_elems)
                self.rank[j][i] = self.rank[i][j]

        self.sm = np.zeros((n, n))
        prefix_sm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                prefix_sm[i][j] = self.rank[i][j]
                if i - 1 >= 0: prefix_sm[i][j] += prefix_sm[i - 1][j]
                if j - 1 >= 0: prefix_sm[i][j] += prefix_sm[i][j - 1]
                if i - 1 >= 0 and j - 1 >= 0: prefix_sm[i][j] -= prefix_sm[i - 1][j - 1]
        for i in range(n):
            for j in range(i, n):
                if i == 0:
                    self.sm[i][j] = prefix_sm[j][j]
                else:
                    self.sm[i][j] = prefix_sm[j][j] - prefix_sm[i - 1][j] \
                                    - prefix_sm[j][i - 1] + prefix_sm[i - 1][i - 1]
                self.sm[j][i] = self.sm[i][j]

        # step 4, determine boundaries
        D = 1.0 * self.sm[0][n - 1] / (n * n)
        darr, region_arr, idx = [D], [_Region(0, n - 1, self.sm)], []
        sum_region, sum_area = float(self.sm[0][n - 1]), float(n * n)
        for i in range(n - 1):
            mx, pos = -1e9, -1
            for j, region in enumerate(region_arr):
                if region.l == region.r:
                    continue
                region.split(self.sm)
                den = sum_area - region.area + region.lch.area + region.rch.area
                cur = (sum_region - region.tot + region.lch.tot + region.rch.tot) / den
                if cur > mx:
                    mx, pos = cur, j
            assert(pos >= 0)
            tmp = region_arr[pos]
            region_arr[pos] = tmp.rch
            region_arr.insert(pos, tmp.lch)
            sum_region += tmp.lch.tot + tmp.rch.tot - tmp.tot
            sum_area += tmp.lch.area + tmp.rch.area - tmp.area
            darr.append(sum_region / sum_area)
            idx.append(tmp.best_pos)

        dgrad = [(darr[i + 1] - darr[i]) for i in range(len(darr) - 1)]

        # C99-style smoothing: convolution with mask {1,2,4,8,4,2,1}
        if len(dgrad) >= 2:
            weights = [1, 2, 4, 8, 4, 2, 1]
            half_w = len(weights) // 2  # 3
            smoothed = []
            
            for i in range(len(dgrad)):
                acc = 0.0
                wsum = 0.0
                for k, w in enumerate(weights):
                    j = i + k - half_w  # center the kernel at i
                    if 0 <= j < len(dgrad):
                        acc += w * dgrad[j]
                        wsum += w
                smoothed.append(acc / wsum if wsum > 0 else dgrad[i])
            
            dgrad = smoothed
        # else: for len(dgrad) < 2, leave as-is

        mu = float(np.mean(dgrad))
        sigma = float(np.std(dgrad))
        cutoff = mu + self.std_coeff * sigma  # std_coeff ~ c in the paper
        assert(len(idx) == len(dgrad))
        above_cutoff_idx = [i for i in range(len(dgrad)) if dgrad[i] >= cutoff]
        if len(above_cutoff_idx) == 0: boundary = []
        else: boundary = idx[:max(above_cutoff_idx) + 1]
        ret = [0 for _ in range(n)]
        for i in boundary:
            ret[i] = 1
            # boundary should not be too close
            for j in range(i - 1, i + 2):
                if j >= 0 and j < n and j != i and ret[j] == 1:
                    ret[i] = 0
                    break
        return np.array([1] + ret[:-1], dtype=np.int64)

class _Region:
    """
    Used to denote a rectangular region of similarity matrix.
    """
    def __init__(self, l, r, sm_matrix):
        assert(r >= l)
        self.tot = sm_matrix[l][r]
        self.l = l
        self.r = r
        self.area = (r - l + 1)**2
        self.lch, self.rch, self.best_pos = None, None, -1

    def split(self, sm_matrix):
        if self.best_pos >= 0:
            return
        if self.l == self.r:
            self.best_pos = self.l
            return
        assert(self.r > self.l)
        mx, pos = -1e9, -1
        for i in range(self.l, self.r):
            carea = (i - self.l + 1)**2 + (self.r - i)**2
            cur = (sm_matrix[self.l][i] + sm_matrix[i + 1][self.r]) / carea
            if cur > mx:
                mx, pos = cur, i
        assert(pos >= self.l and pos < self.r)
        self.lch = _Region(self.l, pos, sm_matrix)
        self.rch = _Region(pos + 1, self.r, sm_matrix)
        self.best_pos = pos


class C99Chunker:
    """
    C99-based text segmenter for generic transcripts
    """
    
    def __init__(self, 
                 embedding_model: Optional[str] = "text-embedding-3-small",
                 use_tf_vectors: bool = False,
                 window: int = 6, 
                 std_coeff: float = 1.2,
                 output_dir: Optional[Path] = None):
        """
        Initialize the C99 chunker for transcript segmentation
        
        Args:
            embedding_model: OpenAI model name for generating utterance embeddings (default: text-embedding-3-small)
                            Set to None when using TF vectors
            use_tf_vectors: If True, use term-frequency vectors (original C99 approach) instead of embeddings
            window: Window size for local similarity ranking in C99 (default: 6, gives 11x11 mask)
            std_coeff: Threshold coefficient for boundary detection (default: 1.2)
            output_dir: Directory to save segmentation results (default: ./output)
        """
        self.embedding_model = embedding_model
        self.use_tf_vectors = use_tf_vectors
        self.c99 = C99(window=window, std_coeff=std_coeff)
        self._embedding_client = None
        self._stemmer = None
        self._stopwords = None
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_embedding_client(self):
        """Lazy initialization of OpenAI embedding client"""
        if self._embedding_client is None:
            try:
                import openai
                import os
                
                # Check if API key is available
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY not found in environment variables. "
                        "Please set your OpenAI API key."
                    )
                
                self._embedding_client = openai.Client(api_key=api_key)
                logger.info(f"Initialized OpenAI client with model: {self.embedding_model}")
                
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise RuntimeError(f"Cannot initialize OpenAI embeddings client: {e}")
        return self._embedding_client
    
    def _get_embeddings(self, texts: List[str]) -> npt.NDArray[np.float64]:
        """
        Generate embeddings for a list of texts using OpenAI API
        
        Args:
            texts: List of utterance texts
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        if not texts:
            return np.array([], dtype=np.float64)
        
        client = self._get_embedding_client()
        
        try:
            # Process in batches to handle rate limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = client.embeddings.create(
                    input=batch_texts,
                    model=self.embedding_model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                batch_num = i // batch_size + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size
                logger.info(f"Processed batch {batch_num}/{total_batches}")
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return np.array(all_embeddings, dtype=np.float64)
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def _get_stemmer_and_stopwords(self):
        """Lazy initialization of stemmer and stopwords for TF vectors"""
        if self._stemmer is None:
            try:
                from nltk.stem.porter import PorterStemmer
                from nltk.corpus import stopwords
                import nltk
                
                # Try to use stopwords, download if needed
                try:
                    self._stopwords = set(stopwords.words("english"))
                except LookupError:
                    logger.info("Downloading NLTK stopwords...")
                    nltk.download('stopwords', quiet=True)
                    self._stopwords = set(stopwords.words("english"))
                
                self._stemmer = PorterStemmer()
                logger.info("Initialized Porter stemmer and stopwords for TF vectors")
                
            except Exception as e:
                logger.error(f"Failed to initialize NLTK components: {e}")
                raise RuntimeError(f"Cannot initialize stemmer/stopwords: {e}")
        
        return self._stemmer, self._stopwords
    
    def _build_tf_vectors(self, texts: List[str]) -> npt.NDArray[np.float64]:
        """
        Build term-frequency vectors as in original C99:
        - lowercase
        - strip punctuation
        - remove stopwords
        - apply Porter stemming
        - count word stems per sentence
        
        Args:
            texts: List of utterance texts
            
        Returns:
            Array of TF vectors with shape (len(texts), vocabulary_size)
        """
        if not texts:
            return np.array([], dtype=np.float64)
        
        stemmer, stop = self._get_stemmer_and_stopwords()
        token_pattern = re.compile(r"[A-Za-z]+")
        
        # 1) Tokenize + normalize + stem
        tokenized = []
        vocab = {}
        for txt in texts:
            tokens = token_pattern.findall(txt.lower())
            stems = [stemmer.stem(t) for t in tokens if t not in stop]
            tokenized.append(stems)
            for s in stems:
                if s not in vocab:
                    vocab[s] = len(vocab)
        
        V = len(vocab)
        logger.info(f"Built vocabulary of {V} unique stems from {len(texts)} utterances")
        
        # Handle edge case of empty vocabulary
        if V == 0:
            logger.warning("Empty vocabulary after stemming and stopword removal")
            return np.zeros((len(texts), 1), dtype=np.float64)
        
        mat = np.zeros((len(texts), V), dtype=np.float64)
        
        # 2) Build frequency vectors
        for i, stems in enumerate(tokenized):
            counts = Counter(stems)
            for s, c in counts.items():
                j = vocab[s]
                mat[i, j] = float(c)
        
        return mat
    
    
    def segment_transcript(self, transcript_data: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[int]]:
        """
        Segment a transcript using C99 algorithm
        
        Args:
            transcript_data: DataFrame with columns 'utterance_id' and 'utterance_text'
            
        Returns:
            Tuple of (segments, boundary_indices)
            - segments: List of DataFrame segments
            - boundary_indices: List of utterance indices where boundaries were detected
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate input
        required_columns = {'utterance_id', 'utterance_text'}
        if not required_columns.issubset(transcript_data.columns):
            missing = required_columns - set(transcript_data.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Starting C99 segmentation for transcript with {len(transcript_data)} utterances")
        
        # Sort by utterance_id to maintain order
        transcript_data = transcript_data.sort_values('utterance_id').copy()
        
        # Extract utterance texts for embedding
        utterance_texts = transcript_data['utterance_text'].tolist()
        
        # Generate vectors (either embeddings or TF vectors)
        if self.use_tf_vectors:
            logger.info(f"Building TF vectors for {len(utterance_texts)} utterances")
            vectors = self._build_tf_vectors(utterance_texts)
        else:
            logger.info(f"Generating embeddings for {len(utterance_texts)} utterances")
            vectors = self._get_embeddings(utterance_texts)
        
        # Apply C99 segmentation
        logger.info("Applying C99 segmentation algorithm")
        boundaries = self.c99.segment(vectors)
        
        # Convert boundary array to segment boundaries
        boundary_indices = [i for i, is_boundary in enumerate(boundaries) if is_boundary]
        logger.info(f"Detected {len(boundary_indices)} topic boundaries at indices: {boundary_indices}")
        
        # Create segments based on boundaries
        segments = []
        transcript_rows = list(transcript_data.itertuples(index=False))
        
        for i, start_idx in enumerate(boundary_indices):
            # Determine end index for this segment
            if i + 1 < len(boundary_indices):
                end_idx = boundary_indices[i + 1]
            else:
                end_idx = len(transcript_rows)
            
            # Extract segment rows
            segment_rows = transcript_rows[start_idx:end_idx]
            if segment_rows:
                segment_df = pd.DataFrame(segment_rows, columns=transcript_data.columns)
                segments.append(segment_df)
        
        logger.info(f"Created {len(segments)} segments")
        
        # Log segment statistics
        segment_sizes = [len(seg) for seg in segments]
        logger.info(f"Segment sizes: {segment_sizes} (avg: {np.mean(segment_sizes):.1f})")
        
        return segments, boundary_indices
    
    def save_results(self, 
                    transcript_name: str,
                    segments: List[pd.DataFrame], 
                    boundary_indices: List[int]) -> Path:
        """
        Save C99 segmentation results to files
        
        Args:
            transcript_name: Name of the transcript (used for output filenames)
            segments: List of segment DataFrames
            boundary_indices: List of boundary indices
            
        Returns:
            Path to the output directory
        """
        # Create output directory for this transcript
        output_path = self.output_dir / transcript_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save segmentation metadata
        metadata = {
            "transcript_name": transcript_name,
            "algorithm": "C99",
            "vector_method": "tf_vectors" if self.use_tf_vectors else "embeddings",
            "embedding_model": self.embedding_model if not self.use_tf_vectors else None,
            "window": self.c99.window,
            "std_coeff": self.c99.std_coeff,
            "num_segments": len(segments),
            "boundary_indices": boundary_indices,
            "segment_sizes": [len(seg) for seg in segments],
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual segments as CSV files
        segments_dir = output_path / "segments"
        segments_dir.mkdir(exist_ok=True)
        
        for seg_idx, segment in enumerate(segments):
            segment_path = segments_dir / f"segment_{seg_idx + 1}.csv"
            segment.to_csv(segment_path, index=False)
        
        # Save full segmentation results as JSON
        all_segments_data = {
            "metadata": metadata,
            "segments": []
        }
        
        for seg_idx, segment in enumerate(segments):
            segment_data = {
                "segment_id": seg_idx + 1,
                "start_utterance_id": int(segment.iloc[0]['utterance_id']),
                "end_utterance_id": int(segment.iloc[-1]['utterance_id']),
                "num_utterances": len(segment),
                "utterances": segment.to_dict('records')
            }
            all_segments_data["segments"].append(segment_data)
        
        full_results_path = output_path / "full_results.json"
        with open(full_results_path, 'w') as f:
            json.dump(all_segments_data, f, indent=2)
        
        logger.info(f"Segmentation results saved to {output_path}")
        logger.info(f"  - Metadata: {metadata_path}")
        logger.info(f"  - Full results: {full_results_path}")
        logger.info(f"  - Individual segments: {len(segments)} files in {segments_dir}")
        
        return output_path
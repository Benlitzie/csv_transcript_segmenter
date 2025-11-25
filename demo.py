#!/usr/bin/env python3
"""
Demo Script for C99 Transcript Segmenter

Interactive script to segment transcripts using the C99 algorithm.
"""

import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.c99_segmenter import C99Chunker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_segment_boundaries(transcript_df: pd.DataFrame, boundary_indices: list, segments: list):
    """Print a visual representation of detected boundaries"""
    print("\n" + "="*80)
    print("SEGMENTATION RESULTS")
    print("="*80)
    
    print(f"\nTotal utterances: {len(transcript_df)}")
    print(f"Number of segments: {len(segments)}")
    print(f"Boundary indices: {boundary_indices}")
    
    print("\n" + "-"*80)
    for i, start_idx in enumerate(boundary_indices):
        if i + 1 < len(boundary_indices):
            end_idx = boundary_indices[i + 1]
        else:
            end_idx = len(transcript_df)
        
        segment_size = end_idx - start_idx
        start_utt_id = transcript_df.iloc[start_idx]['utterance_id']
        end_utt_id = transcript_df.iloc[end_idx - 1]['utterance_id']
        
        print(f"\nSEGMENT {i + 1}: Utterances {start_utt_id}-{end_utt_id} ({segment_size} utterances)")
        print("-"*80)
        
        # Show first 3 utterances
        segment = segments[i]
        for j, (_, row) in enumerate(segment.head(3).iterrows()):
            text = row['utterance_text']
            if len(text) > 70:
                text = text[:67] + "..."
            print(f"  {row['utterance_id']:2d}. {text}")
        
        if len(segment) > 3:
            print(f"  ... [{len(segment) - 3} more utterances]")
    
    print("\n" + "="*80)


def get_user_input():
    """Get segmentation parameters from user"""
    print("\n" + "="*80)
    print("C99 TRANSCRIPT SEGMENTER")
    print("="*80)
    
    # Get transcript path
    print("\nEnter the path to your transcript CSV file:")
    print("(Press Enter for default: data/demo.csv)")
    transcript_path = input("> ").strip()
    if not transcript_path:
        transcript_path = "data/demo.csv"
    
    # Choose vector method
    print("\nChoose vector representation method:")
    print("  1. OpenAI Embeddings (modern, requires API key)")
    print("  2. TF Vectors (original C99 paper approach, requires NLTK)")
    print("(Press Enter for default: 1)")
    method_input = input("> ").strip()
    if method_input == "2":
        use_tf_vectors = True
        embedding_model = None
    else:
        use_tf_vectors = False
        # Get embedding model
        print("\nEnter OpenAI embedding model:")
        print("(Press Enter for default: text-embedding-3-small)")
        embedding_model = input("> ").strip()
        if not embedding_model:
            embedding_model = "text-embedding-3-small"
    
    # Get window size
    print("\nEnter window size (recommended: 5-15, paper default: 6 for 11x11 mask):")
    print("(Press Enter for default: 6)")
    window_input = input("> ").strip()
    if window_input:
        try:
            window = int(window_input)
        except ValueError:
            print("Invalid input. Using default: 6")
            window = 6
    else:
        window = 6
    
    # Get std_coeff
    print("\nEnter threshold coefficient (paper default: 1.2):")
    print("(Press Enter for default: 1.2)")
    std_coeff_input = input("> ").strip()
    if std_coeff_input:
        try:
            std_coeff = float(std_coeff_input)
        except ValueError:
            print("Invalid input. Using default: 1.2")
            std_coeff = 1.2
    else:
        std_coeff = 1.2
    
    # Get output directory
    print("\nEnter output directory:")
    print("(Press Enter for default: output)")
    output_dir = input("> ").strip()
    if not output_dir:
        output_dir = "output"
    
    return transcript_path, use_tf_vectors, embedding_model, window, std_coeff, output_dir


def main():
    """Main demo function"""
    
    # Load environment variables (for OPENAI_API_KEY)
    load_dotenv()
    
    # Get user input
    transcript_path, use_tf_vectors, embedding_model, window, std_coeff, output_dir = get_user_input()
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Transcript: {transcript_path}")
    print(f"Vector Method: {'TF Vectors (original C99)' if use_tf_vectors else 'OpenAI Embeddings'}")
    if not use_tf_vectors:
        print(f"Embedding Model: {embedding_model}")
    print(f"Window: {window}")
    print(f"Std Coeff: {std_coeff}")
    print(f"Output Directory: {output_dir}")
    print("="*80)
    
    # Validate input file
    transcript_path = Path(transcript_path)
    if not transcript_path.exists():
        logger.error(f"Transcript file not found: {transcript_path}")
        return 1
    
    # Load transcript
    logger.info(f"Loading transcript from {transcript_path}")
    try:
        transcript_df = pd.read_csv(transcript_path)
    except Exception as e:
        logger.error(f"Failed to load transcript: {e}")
        return 1
    
    # Validate columns
    required_columns = {'utterance_id', 'utterance_text'}
    if not required_columns.issubset(transcript_df.columns):
        missing = required_columns - set(transcript_df.columns)
        logger.error(f"Missing required columns: {missing}")
        return 1
    
    # Remove rows with missing text
    transcript_df = transcript_df.dropna(subset=['utterance_text'])
    logger.info(f"Loaded {len(transcript_df)} utterances")
    
    # Initialize segmenter
    logger.info(f"Initializing C99 Segmenter (window={window}, std_coeff={std_coeff})")
    chunker = C99Chunker(
        embedding_model=embedding_model,
        use_tf_vectors=use_tf_vectors,
        window=window,
        std_coeff=std_coeff,
        output_dir=Path(output_dir)
    )
    
    # Segment transcript
    logger.info("Starting segmentation...")
    try:
        segments, boundary_indices = chunker.segment_transcript(transcript_df)
        
        # Print results
        print_segment_boundaries(transcript_df, boundary_indices, segments)
        
        # Save results
        transcript_name = transcript_path.stem
        output_path = chunker.save_results(
            transcript_name=transcript_name,
            segments=segments,
            boundary_indices=boundary_indices
        )
        
        print(f"\nResults saved to: {output_path}")
        print(f"  - metadata.json: Configuration and statistics")
        print(f"  - full_results.json: Complete segmentation results")
        print(f"  - segments/: Individual segment CSV files")
        
        logger.info("Segmentation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

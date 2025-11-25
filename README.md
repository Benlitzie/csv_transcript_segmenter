# C99 Transcript Segmenter

A Python implementation of the C99 algorithm for automatic text segmentation. This tool segments transcripts into topically coherent segments based on semantic similarity using OpenAI embeddings.

## Features

- **Automatic Topic Segmentation**: Uses the C99 algorithm to detect topic boundaries in transcripts
- **Semantic Embeddings**: Leverages OpenAI's text embedding models for deep semantic understanding
- **Flexible Output**: Saves results in both JSON and CSV formats with metadata
- **CLI Interface**: Simple command-line interface with customizable parameters

## Installation

1. Install dependencies using Poetry:
```bash
poetry install --no-root
```

2. Set up your OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Quick Start

Run the interactive demo script:

```bash
poetry run python demo.py
```

The script will prompt you to enter:
1. Path to your transcript CSV file (default: `data/demo.csv`)
2. Window size (default: `10`)
3. Threshold coefficient (default: `1.0`)
4. Output directory (default: `output`)

Simply press Enter to use the default values.

## Programmatic Usage

```python
from pathlib import Path
import pandas as pd
from src.c99_segmenter import C99Chunker

# Load your transcript (must have 'utterance_id' and 'utterance_text' columns)
transcript_df = pd.read_csv("your_transcript.csv")

# Initialize the segmenter
chunker = C99Chunker(
    embedding_model="text-embedding-3-small",
    window=10,           # Window size for local similarity ranking
    std_coeff=1.0,       # Threshold coefficient for boundary detection
    output_dir=Path("output")
)

# Segment the transcript
segments, boundary_indices = chunker.segment_transcript(transcript_df)

# Save results
chunker.save_results(
    transcript_name="my_transcript",
    segments=segments,
    boundary_indices=boundary_indices
)
```

## Input Format

Your transcript CSV must have these columns:

- `utterance_id`: Sequential ID for each utterance (integer)
- `utterance_text`: The text content of each utterance (string)

Example:

```csv
utterance_id,utterance_text
1,"Good morning! How are you feeling today?"
2,"Hi, thank you for asking. I've been having a tough week."
3,"I'm sorry to hear that. Can you tell me more?"
```

## Output Structure

Results are saved in the following structure:

```
output/
└── transcript_name/
    ├── metadata.json           # Segmentation configuration and stats
    ├── full_results.json        # Complete results with all segments
    └── segments/
        ├── segment_1.csv
        ├── segment_2.csv
        └── segment_N.csv
```

### metadata.json

Contains configuration parameters and segmentation statistics:

```json
{
  "transcript_name": "demo",
  "algorithm": "C99",
  "embedding_model": "text-embedding-3-small",
  "window": 10,
  "std_coeff": 1.0,
  "num_segments": 5,
  "boundary_indices": [0, 12, 23, 35, 42],
  "segment_sizes": [12, 11, 12, 7, 6],
  "timestamp": "2025-11-21T10:30:00"
}
```

### full_results.json

Contains complete segmentation results with all utterances:

```json
{
  "metadata": {...},
  "segments": [
    {
      "segment_id": 1,
      "start_utterance_id": 1,
      "end_utterance_id": 12,
      "num_utterances": 12,
      "utterances": [...]
    }
  ]
}
```

## Algorithm Parameters

### window (default: 10)
Window size for local similarity ranking. Larger values create smoother similarity matrices but may miss fine-grained boundaries.

### std_coeff (default: 1.0)
Threshold coefficient for boundary detection. Higher values result in fewer segments (only strongest boundaries), lower values result in more segments.

## How C99 Works

1. **Embedding Generation**: Each utterance is converted to a semantic embedding vector
2. **Similarity Matrix**: Computes cosine similarity between all utterance pairs
3. **Rank Matrix**: Applies local ranking within sliding windows
4. **Dynamic Programming**: Finds optimal segmentation by maximizing intra-segment similarity
5. **Boundary Detection**: Identifies topic boundaries using gradient analysis

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies: numpy, pandas, openai, python-dotenv

## License

This implementation is based on the C99 algorithm from:
> Freddy Y. Y. Choi. 2000. Advances in domain independent linear text segmentation.
> In Proceedings of NAACL 2000.

Original algorithm reference: https://github.com/logological/C99

## Parameter Tuning

### window (default: 6)
Window size for local similarity ranking. Larger values create smoother similarity matrices but may miss fine-grained boundaries.

- **Smaller values**: More sensitive to local changes, may create more segments
- **Larger values**: Considers broader context, may create fewer segments

### std_coeff (default: 1.2)
Threshold coefficient for boundary detection. Higher values result in fewer segments (only strongest boundaries), lower values result in more segments.

- **Lower values**: More segments (detects weaker boundaries)
- **Higher values**: Fewer segments (only strong boundaries)

**Tip**: Start with default values (6, 1.2) and adjust based on results.

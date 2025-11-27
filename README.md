# Homeless Sentiment Analysis

This project analyzes Canadian homelessness-related Reddit posts for sentiment and emotion detection.

## Prerequisites

### Install uv

`uv` is a fast Python package installer and resolver. Install it on your operating system:

#### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, you may need to restart your terminal or add uv to your PATH.

## Getting Started

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone <repository-url>
   cd Homeless_Sentiment
   ```

2. **Sync dependencies** using uv:

   ```bash
   uv sync
   ```

   This command will:
   - Create a virtual environment
   - Install all required dependencies from `pyproject.toml`
   - Lock dependencies for reproducible builds

3. **Run the sentiment analysis pipeline**:

   Open the Jupyter notebook `sentiment_emotion_pipeline.ipynb` in your preferred environment:

   - **VS Code**: Open the notebook file and select the kernel created by uv
   - **Jupyter Lab/Notebook**:

     ```bash
     uv run jupyter lab
     ```

     Then navigate to `sentiment_emotion_pipeline.ipynb` and run the cells

## Project Structure

- `canadian_homelessness_reddit_posts.csv` - Raw Reddit post data
- `reddit_posts_with_sentiment_emotion.csv` - Processed data with sentiment/emotion labels
- `sentiment_emotion_pipeline.ipynb` - Main analysis notebook
- `redditCrawl.ipynb` - Data collection notebook
- `main.py` - Python script version
- `pyproject.toml` - Project dependencies and configuration
- `requirements.txt` - Legacy requirements file

## Usage

Run all cells in `sentiment_emotion_pipeline.ipynb` sequentially to:

1. Load the Reddit posts data
2. Apply sentiment analysis
3. Detect emotions in the posts
4. Generate visualizations and insights

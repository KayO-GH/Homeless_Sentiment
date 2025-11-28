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

## Output Files

The notebook generates `reddit_posts_with_sentiment_emotion.csv` with the following columns:

### Basic Information

- **id** - Unique Reddit post identifier
- **city** - Canadian city associated with the post
- **title** - Original post title
- **selftext** - Original post body text
- **combined_text** - Concatenated title and selftext used for analysis
- **score** - Reddit post score (upvotes - downvotes)
- **num_comments** - Number of comments on the post
- **created_utc** - Post creation timestamp
- **url** - Direct link to the Reddit post

### Sentiment Analysis Results

- **sentiment_consensus** - Final sentiment prediction based on majority vote from all models (`positive`, `negative`, or `neutral`)
- **sentiment_roberta_label** - Sentiment from Twitter-RoBERTa model
- **sentiment_roberta_score** - Confidence score for Twitter-RoBERTa prediction (0-1)
- **sentiment_siebert_label** - Sentiment from SiEBERT model
- **sentiment_siebert_score** - Confidence score for SiEBERT prediction (0-1)
- **sentiment_bertweet_label** - Sentiment from BERTweet model
- **sentiment_bertweet_score** - Confidence score for BERTweet prediction (0-1)

### Emotion Analysis Results

- **emotion_roberta_primary** - Primary emotion from Twitter-RoBERTa emotion model
- **emotion_roberta_primary_score** - Confidence score for primary emotion (0-1)
- **emotion_distilroberta_primary** - Primary emotion from DistilRoBERTa model (7 basic emotions)
- **emotion_distilroberta_primary_score** - Confidence score for primary emotion (0-1)
- **emotion_multilabel_labels** - Comma-separated list of emotions detected by multi-label model
- **emotion_goemotions_top3** - Top 3 emotions from GoEmotions model (Reddit-specific, 28 emotions)

### Models Used

**Sentiment Analysis (3 models):**

- Twitter-RoBERTa (CardiffNLP) - Social media optimized
- SiEBERT (RoBERTa-large) - General purpose
- BERTweet - Social media specific

**Emotion Classification (4 models):**

- Twitter-RoBERTa (single-label) - Basic emotions
- Twitter-RoBERTa (multi-label) - Multiple emotions per post
- GoEmotions - Reddit-specific with 28 emotion categories
- DistilRoBERTa - 7 basic emotions (6 Ekman + neutral)

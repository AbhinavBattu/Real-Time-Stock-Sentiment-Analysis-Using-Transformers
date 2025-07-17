# Reddit Stock Sentiment Analyzer 📈

A real-time stock sentiment analysis tool that fetches Reddit posts about stocks, analyzes sentiment using a custom Transformer model, and provides actionable trading recommendations.

![UI Screenshot1](https://github.com/AbhinavBattu/Real-Time-Stock-Sentiment-Analysis-Using-Transformers/blob/main/images/Pic1.png) <!-- Replace with actual image path -->

![UI Screenshot2](https://github.com/AbhinavBattu/Real-Time-Stock-Sentiment-Analysis-Using-Transformers/blob/main/images/Pic2.png)

## 🌟 Features

- Real-time Reddit post fetching for any stock
- Custom-trained Transformer-based sentiment analysis model
- Clear BUY/HOLD/SELL recommendations
- Simple API backend with FastAPI
- User-friendly Streamlit interface

## 🛠️ Technologies Used
- Python 3.10
- FastAPI (API framework)
- PyTorch (Deep Learning)
- Feedparser (RSS parsing)
- Streamlit (Web UI)
- Plotly (Visualizations)

# Reddit Stock Sentiment Analyzer

This project provides real-time stock recommendations by analyzing sentiment from Reddit posts related to specific companies. It leverages a custom-built Transformer model for robust sentiment analysis.

## How It Works

Our system provides real-time stock recommendations by leveraging the collective sentiment expressed on Reddit. Here's a breakdown of the process:

1.  **Data Collection**: The system initiates by fetching real-time posts from Reddit that are highly relevant to the input stock name. This ensures we capture the most current public discourse surrounding a given company.

2.  **Sentiment Analysis**: Each collected post undergoes rigorous processing through our custom-built **Transformer Classifier model**. This model, an advanced neural network architecture, is specifically designed for sequential data like text.

    * **Training**: We train our Transformer model on a labeled dataset of Reddit posts, learning to associate linguistic patterns with positive or negative sentiment.

       At its core, the **Transformer Classifier** employs:
       * **Embedding Layer**: Converts input text (words) into dense vector representations, capturing semantic meaning.
       * **Positional Encoding**: Since Transformers process words in parallel without inherent sequence understanding, our custom `PositionalEncoding` module injects information about the relative or absolute position of words in the input sequence using sine and cosine functions. This ensures the model understands word order, which is crucial for sentiment.
       * **Transformer Encoder**: This is the powerhouse of the model, consisting of multiple layers. Each layer contains multi-head self-attention mechanisms, allowing the model to weigh the importance of different words in a sentence relative to each other, and feed-forward neural networks. This architecture excels at capturing long-range dependencies and contextual relationships within the text.
       * **Classification Head**: A final set of linear layers with a ReLU activation and dropout for regularization, followed by a softmax function, processes the aggregated output from the Transformer Encoder to predict the sentiment (positive or negative).
  
    * **Inference**:
      - The trained Transformer model processes real-time Reddit posts, classifying each as positive or negative to generate an overall sentiment score.
      - The model utilizes **pre-trained weights**, meaning it has already learned intricate patterns and nuances of language from a vast dataset, enabling highly accurate inference on new, unseen Reddit posts.        - Based on this inference, each post is labeled as either "positive" or "negative," which then contributes to an overall sentiment score.

3.  **Recommendation Engine**: The system aggregates the sentiment scores from all analyzed posts to generate a comprehensive recommendation:
    * **Positive Score > 0.2**: A strong positive sentiment across Reddit posts indicates a favorable outlook, leading to a **BUY** recommendation.
    * **Negative Score < -0.2**: A significant negative sentiment suggests potential concerns or a bearish outlook, resulting in a **DON'T BUY** recommendation.
    * **Between -0.2 and 0.2**: When sentiment is relatively balanced or neutral, the system advises to **HOLD**, suggesting neither strong positive nor negative indicators are present.

This robust framework ensures that our stock recommendations are not just based on raw data, but on a deep, context-aware understanding of public sentiment, powered by state-of-the-art deep learning.

## API

The system exposes a simple API endpoint for stock sentiment prediction using fastAPI. [`POST /predict`]

## Getting Started

Follow these instructions to get a local copy of the project up and running.

### Prerequisites
- Python 3.10
- pip package manager

### ⚙️ Installation & Usage

1.  **Clone the Repository**

    Clone the project from GitHub to your local machine.
    ```bash
    git clone [https://github.com/your-username/reddit-stock-sentiment.git](https://github.com/your-username/reddit-stock-sentiment.git)
    ```

2.  **Navigate to the Project Directory**

    Move into the folder that was just created.
    ```bash
    cd reddit-stock-sentiment
    ```
    *Reason: All subsequent commands must be run from the project's root directory.*

3.  **Install Required Libraries**

    Install all the Python packages the project depends on.
    ```bash
    pip install -r requirements.txt
    ```
    *Reason: This command reads the `requirements.txt` file and automatically installs the specific versions of libraries (like FastAPI, Streamlit, scikit-learn, etc.) needed to run the application.*

4.  **Train the Sentiment Model**

    Run the training script to build the machine learning model.
    ```bash
    python train/train.py
    ```
    *Reason: The sample training data is available in the csv format in the data/sentiment.csv. You can also replace it with data you want to train it on*

5.  **Launch the Application**


    **A: Launch the FastAPI Backend API** ⚙️

    ```bash
    uvicorn api:app --reload
    ```
    *Reason: This starts a local web server (using Uvicorn) to serve the model as an API. It's accessible at `http://127.0.0.1:8000`. The `--reload` flag is useful for development as it automatically restarts the server when code changes are detected.*

    **B: Launch the Streamlit Web App** 🎈

    ```bash
    streamlit run app.py
    ```
    *Reason: This command starts the user-friendly graphical interface built with Streamlit. You can interact with the model directly in your web browser.At `http://localhost:8501/`*

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

## How It Works

1. **Data Collection**: The system fetches real-time posts from Reddit related to the input stock name.
2. **Sentiment Analysis**: Each post is processed through our custom Transformer model. And the already trained weights are used to do inference on them and label them as positive and negative then from which we calculate the score and give recommendation accordingly
3. **Recommendation Engine**: Aggregates scores to provide:
   - Positive score > 0.2 → BUY
   - Negative score < -0.2 → DON'T BUY
   - Between → HOLD

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

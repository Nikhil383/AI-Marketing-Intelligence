# Talknlock Marketing Recommendation System

An AI-powered marketing recommendation system that analyzes marketing content and provides actionable insights to improve engagement and performance.

---

## key features

- **Content Analysis**: Analyze marketing content based on format, platform, and engagement score.
- **Actionable Recommendations**: Receive concise, bulleted execution tips to enhance marketing strategies.

## Project Structure

```text
talknlock_assignment/
├── src/
│   ├── app.py                # Main application logic
│   ├── config.py             # Configuration settings 
│   ├── generate.py           # Model generation functionality
|   └── preprocessor.py       # Data preprocessing utilities
│   ├── config.py             # Configuration settings 
│   ├── generate.py           # Model generation functionality
|   └── preprocessor.py       # Data preprocessing utilities
|   |__ train.py              # Model training script
|   |__ mlops.py              # MLOps utilities for deployment
|   |__ .env                  # Environment variables for API keys
├── data/
|   └── marketing_data.csv     # Sample marketing data
|   └── recommendation_model.pkl # Pre-trained recommendation model
|   └── README.md              # Project documentation
```
## Installation

1. Clone the repository:
   ```bash
   git clone

    ```
2. Navigate to the project directory:
    ```bash
    cd talknlock_assignment
    ```
3. Install the required dependencies:
    ```bash
    uv sync
    ```
4. Set up environment variables by creating a `.env` file in the `src/` directory and adding your Gemini API key:
    ```bash
    GEMINI_API_KEY=your_api_key_here
    ```
5. Run the application:
    ```bash
    uv run streamlit run src/app.py
    ```
## Technologies Used

- **Python**: Programming language for application logic.
- **Streamlit**: Framework for building interactive web applications.

## Dataset Description

The dataset contains sample marketing data including information about content format, platform, and engagement metrics.

## Run the Application

- To generate the synthetic dataset, run the `generate.py` script:

  ```bash
    uv run python src/generate.py
  ```

- To preprocess the data, run the `preprocessor.py` script:
  
    ```bash
    uv run python src/preprocessor.py
    ```

- To train the recommendation model, run the `train.py` script:

    ```bash
    uv run python src/train.py
    ```

- To deploy the application, run the `mlops.py` script:

    ```bash
    uv run python src/mlops.py

    ```

- To run the application, execute the main script:

    ```bash
    uv run streamlit run src/app.py
    ```

## Demo


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

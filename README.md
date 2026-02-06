# Heart Disease Prediction System 🫀


## 📌 Overview

The **Heart Disease Prediction System** is an end-to-end Machine Learning solution designed to assess the likelihood of heart disease in patients based on medical parameters. This project aims to assist medical professionals by providing a rapid, automated, and accurate diagnostic support tool, reducing the delay associated with manual risk assessment.

The system is built with a modular, production-ready architecture, ensuring scalability, maintainability, and ease of deployment.

## ✨ Key Features

*   **Modular "Production-Grade" Architecture**: Organized codebase separating data ingestion, transformation, and model training logic.
*   **Automated Data Ingestion**: Robust handling of raw CSV data with automatic train/test splitting.
*   **Advanced Data Transformation**: Comprehensive preprocessing pipelines including:
    *   Standard scaling for numerical features.
    *   One-Hot Encoding for categorical variables.
    *   Handling of missing values and outliers.
*   **Multi-Model Training**: Evaluated multiple algorithms (Random Forest, Logistic Regression, XGBoost, etc.).
    *   **Best Model**: XGBoost/SVC achieving **~85% Accuracy**.
*   **Interactive Web Interface**: User-friendly frontend built with **Streamlit** for real-time predictions.

## 🛠️ Tech Stack

*   **Language**: Python 3.8+
*   **Machine Learning**: Scikit-Learn, XGBoost
*   **Data Manipulation**: Pandas, NumPy
*   **Web Framework**: Streamlit
*   **Utilities**: Joblib, Logging, Dataclasses

## 📂 Project Structure

```bash
DiseasePredictionSystem/
├── artifacts/              # Generated files (CSV splits, Model files)
│   ├── train.csv
│   ├── test.csv
│   ├── data.csv
│   └── (model.pkl, preprocessor.pkl)
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned data
├── logs/                   # Execution logs
├── notebooks/              # Jupyter notebooks for EDA
├── src/                    # Source code
│   ├── components/         # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/           # Prediction pipeline
│   │   └── predict_pipeline.py
│   ├── exception.py        # Custom exception handling
│   ├── logger.py           # Logging configuration
│   └── utils.py            # Utility functions
├── app.py                  # Streamlit Web Application
├── main.py                 # Training pipeline entry point
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🚀 Installation & Usage

Follow these steps to set up the project locally.

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/DiseasePredictionSystem.git
cd DiseasePredictionSystem
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate
# Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Training Pipeline
Execute the main script to ingest data, transform it, and train the model.
```bash
python main.py
```
*Check the `artifacts/` folder for the saved model and preprocessor.*

### Step 5: Run the Web Application
Launch the Streamlit app to test predictions interactively.
```bash
streamlit run app.py
```

## 📊 Results

The model has been evaluated on a held-out test set and achieves valid performance metrics:

*   **Accuracy**: 85.33%
*   **Robustness**: Handles various input types (categorical/numerical) gracefully via the transformation pipeline.

## 🔮 Future Improvements

*   **Deployment**: Dockerize the application and deploy to AWS EC2 or Azure App Service.
*   **CI/CD**: specific GitHub Actions for automated testing and linting.
*   **Dataset Expansion**: Incorporate larger datasets (e.g., Cleveland, Hungary, Switzerland combined) for better generalization.
*   **Model Monitoring**: Implement tools like MLflow or evidently.ai for drift detection.

---


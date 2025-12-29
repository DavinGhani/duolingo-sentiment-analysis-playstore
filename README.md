# NLP Project: Duolingo App Reviews Sentiment Analysis 🦉

This project focuses on Natural Language Processing (NLP) to perform sentiment analysis on user reviews of the **Duolingo** application from the Google Play Store. The goal is to classify Indonesian reviews into Positive, Negative, or Neutral sentiments using various Machine Learning models.

---

## 📝 Project Description

This project covers the complete end-to-end Machine Learning workflow, starting from data scraping, text preprocessing, lexicon-based labeling, feature engineering, model building, hyperparameter tuning, and inference. The project compares the performance of **Support Vector Machine (SVM)** and **Random Forest** classifiers using different feature extraction techniques (**TF-IDF** and **Word2Vec**).

**Objective**: To build a robust model capable of accurately classifying the sentiment of Indonesian text reviews for the Duolingo app, handling class imbalance using SMOTE.

---

## 📊 Dataset

The dataset was generated dynamically by scraping the Google Play Store.

- **Source**: Google Play Store Reviews (via `google-play-scraper`).
- **Target App**: Duolingo.
- **Language**: Indonesian (`id`).
- **Total Data Scraped**: 15,000 raw reviews.
- **Data Cleaning**: Reduced to **12,191 rows** after removing duplicates and empty comments during preprocessing.
- **Labels**: Generated using a Lexicon-based approach (Positive, Negative, Neutral).

---

## 🚀 Project Workflow

The project is executed in systematic stages:

1.  **Data Acquisition (Scraping)**:
    - Utilized `google-play-scraper` to fetch the latest 15,000 reviews from the Play Store.
    - Exported raw data to CSV format.

2.  **Text Preprocessing**:
    - **Cleaning**: Removal of URLs, mentions, hashtags, numbers, and punctuation.
    - **Case Folding**: Converting text to lowercase.
    - **Tokenization**: Splitting sentences into words.
    - **Stopword Removal**: Using NLTK (Indonesian/English) and a custom stopword list (e.g., 'yg', 'ga', 'sih').
    - **Stemming**: Reducing words to their base form using the **Sastrawi** library.

3.  **Labeling**:
    - Implemented a Lexicon-based sentiment analysis using positive and negative word dictionaries.
    - Scoring logic: Score > 0 (Positive), Score < 0 (Negative), Score = 0 (Neutral).

4.  **Feature Engineering**:
    - **TF-IDF**: Term Frequency-Inverse Document Frequency (Max features: 5000).
    - **Word2Vec**: Word Embeddings using Gensim (Vector size: 100, Window: 5).

5.  **Model Development & Balancing**:
    - Addressed class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.
    - Trained three baseline configurations:
        1.  SVM + TF-IDF (80/20 Split).
        2.  Random Forest + Word2Vec (80/20 Split).
        3.  Random Forest + TF-IDF (70/30 Split).

6.  **Evaluation & Tuning**:
    - Evaluated using Accuracy, Precision, Recall, and F1-Score.
    - Performed **GridSearchCV** to find optimal hyperparameters for both SVM and Random Forest.

---

## 🤖 Model Architectures

The project experimented with the following configurations:

- **Support Vector Machine (SVM)**:
    - Kernel: Linear / RBF.
    - Regularization (C): Tuned via GridSearch.
- **Random Forest**:
    - Estimators: 100, 200.
    - Max Depth: 10, 20, None.
    - Feature extraction input: Both TF-IDF vectors and averaged Word2Vec embeddings.

---

## 📈 Results

After hyperparameter tuning, the **SVM model using TF-IDF features** yielded the best performance.

| Model Configuration | Accuracy | F1-Score (Weighted) | Best Parameters |
| :--- | :--- | :--- | :--- |
| **SVM Tuned (TF-IDF)** | **92.09%** | **0.92** | `{'C': 10, 'kernel': 'linear'}` |
| RF Tuned (TF-IDF) | 88.38% | 0.88 | `{'n_estimators': 200, 'min_samples_leaf': 1}` |
| RF Tuned (Word2Vec) | 77.45% | 0.78 | `{'n_estimators': 200, 'min_samples_leaf': 1}` |

*Conclusion:* The SVM model with a Linear Kernel and TF-IDF feature extraction proved to be the most effective for this specific text classification task.

---

## ⚙️ How to Run the Project

Follow these steps to replicate the analysis.

### Prerequisites
- **Google Account**: To access Google Colab (recommended).
- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `sastrawi`, `gensim`, `scikit-learn`, `imblearn`, `google-play-scraper`.

### Execution Steps

1.  **Install Dependencies**
    Ensure all necessary libraries are installed. You can run this in your environment:
    ```bash
    pip install pandas numpy matplotlib seaborn nltk sastrawi gensim scikit-learn imbalanced-learn google-play-scraper wordcloud
    ```

2.  **Step 1: Data Scraping**
    - Open `Scrapping_Review_Ulasan_Duolingo_Playstore.ipynb`.
    - Run the cells to scrape data from the Play Store.
    - This will generate a file named `reviews_playstore_com.duolingo_15000.csv`.

3.  **Step 2: Sentiment Analysis & Modeling**
    - Open `Sentimen_Analisis_Review_Duolingo_pada_Play_Store.ipynb`.
    - Upload the CSV file generated in Step 1.
    - Run the notebook cells sequentially to perform preprocessing, training, and evaluation.

4.  **Inference (Testing New Reviews)**
    - The final section of the analysis notebook contains an interactive inference loop.
    - You can input any Indonesian sentence, and the trained SVM model will predict whether it is **Positive**, **Negative**, or **Neutral**.

    ```text
    Input: "Duolingo sangat membantu saya belajar bahasa."
    Prediction: 🔥 POSITIF 🔥
    ```

---

# 📧 Spam Detection using Machine Learning

## 📌 Project Overview

This project is a Spam Email Classification System built using:

- Python
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression

The model classifies messages into:

- **Ham (0)** → Not Spam  
- **Spam (1)** → Spam Message  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NLTK  
- Scikit-learn  
- Regular Expressions (re)  

---

## 📂 Dataset

The dataset used is:

**spam.csv**

It contains two main columns:

- `v1` → Label (`ham` or `spam`)
- `v2` → Message text

The labels are converted to:


ham → 0
spam → 1


---

## ⚙️ How the System Works

### 1️⃣ Text Preprocessing

Messages are cleaned using:

- Removing special characters
- Converting text to lowercase
- Removing stopwords
- Stemming words (Porter Stemmer)

Example:

Input:

Congratulations! You've won a free prize!


After preprocessing:

congratul won free prize


---

### 2️⃣ Text Vectorization

The cleaned text is converted into numerical format using:

```python
TfidfVectorizer(max_features=10000)

This converts text into feature vectors that the model can understand.

3️⃣ Train-Test Split

The dataset is split into:

80% Training Data

20% Testing Data

train_test_split(test_size=0.2, random_state=42)
4️⃣ Model Used

The machine learning model used is:

LogisticRegression()

It is a binary classification algorithm.

5️⃣ Model Evaluation

The model performance is evaluated using:

Accuracy Score

Classification Report (Precision, Recall, F1-score)

🚀 How to Run the Project
Step 1: Install Required Libraries
pip install pandas nltk scikit-learn
Step 2: Run the Python Script
python spam_detection.py
Step 3: Enter a Message

The program will ask:

Enter a message to classify as spam or ham:

Example Input:

Congratulations! You have won ₹10,000. Click here now!

Example Output:

The message is spam
📊 Example Output
accuracy: 97.85%
classification report:
              precision    recall  f1-score   support
📈 Model Pipeline
Raw Text
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression Model
   ↓
Spam / Ham Prediction

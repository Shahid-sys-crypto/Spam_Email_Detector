import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")
stemmer=PorterStemmer()
stop_words=set(stopwords.words("english"))

df=pd.read_csv("spam.csv",encoding="latin-1")[["v1","v2"]]
df.columns=["label","text"]
df["label"]=df["label"].map({"ham":0,"spam":1})

def preprocess_text(text):
    text=re.sub(r"\W"," ",text)
    text=text.lower()
    words=text.split()
    words=[stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_text"]=df["text"].apply(preprocess_text)

vectorizer=TfidfVectorizer(max_features=10000)
x=vectorizer.fit_transform(df["cleaned_text"])
y=df["label"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(f"accuracy:,{accuracy_score(y_test,y_pred)*100:.2f}%")
print("classification report:\n",classification_report(y_test,y_pred))

def predict_email(text):
    cleaned_text=preprocess_text(text)
    vectorized_text=vectorizer.transform([cleaned_text])
    prediction=model.predict(vectorized_text)
    return "spam" if prediction[0]==1 else "ham"

def main():
    email=input("enter an message to classify as spam or ham:")
    result=predict_email(email)
    print(f"the message is {result}")

if __name__=="__main__":
    main()
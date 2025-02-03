import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
df=pd.read_csv('./training.csv')
def preprocessText(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]', '', text)
    words=nltk.word_tokenize(text)
    stopWords=set(stopwords.words('english'))
    words=[word for word in words if word not in stopWords]
    lemmatizer=WordNetLemmatizer()
    words=[lemmatizer.lemmatize(word) for word in words]
    return "".join(words)


texts=df['text'].array
labels=df['label'].array

vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(texts)
X_dense=X.todense()
processed_texts = [preprocessText(text) for text in texts]
X_train, X_test, y_train, y_test = train_test_split(processed_texts, labels, test_size=0.3, random_state=42)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

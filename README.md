# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Read the dataset Load the spam.csv file and select the message and label columns.
 2.Preprocess the data Convert labels (ham, spam) into numeric values (0, 1) and transform text messages into numerical features using TF-IDF  3.Split the dataset Divide the data into training data and testing data using train_test_split(). 
 4.Train the SVM model Create the SVM classifier and train it using the training dataset. 
 5.Predict and evaluate Predict spam/ham messages for test data and display the confusion matrix using a heatmap graph.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
/*
Program to implement the SVM For Spam Mail Detection..
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

data['label'] = data['label'].map({'ham':0, 'spam':1})

X = data['message']
y = data['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)


model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Spam Detection")
plt.show() 
Developed by: YAZHINI K
RegisterNumber:  212225220126
*/
```

## Output:
![SVM For Spam Mail Detection](sam.png)
<img width="667" height="487" alt="WhatsApp Image 2026-05-26 at 6 39 13 PM" src="https://github.com/user-attachments/assets/1867efb7-8b64-45f2-aa0e-6c5690643037" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

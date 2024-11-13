# 4. Assignment : Bays Classifier
# Problem Statement: A SMS unsolicited mail (every now and then known
as cell smartphone junk
# mail) is any junk message brought to a cellular phone as textual
content messaging via the Short
# Message Service (SMS). Use probabilistic approach (Naive Bayes
Classifier / Bayesian Network)
# to implement SMS Spam Filtering system. SMS messages are categorized
as SPAM or HAM
# using features like length of message, word depend, unique keywords
etc. Download Data -Set
# from :http://archive.ics.uci.edu/ml/datasets/sms+spam+collection
# This dataset is composed by just one text file, where each line has
the correct class followed by
# the raw message
# Apply Machine Learning Algorithm and Evaluate Model
import pandas as pd
df = pd.read_csv("/content/SMSSpamCollection", delimiter="\t",
header=None)
df.head(n=10)
df.columns = ["Class_label", "Message"]
df["Class_label"] = df["Class_label"].apply(lambda x: 1 if x == "spam"
else 0)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
 df["Message"], df["Class_label"], test_size=0.3, random_state=0
)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(df["Message"])
print(vectorizer.vocabulary_)
new_train_set = vectorizer.transform(x_train)
new_test_set = vectorizer.transform(x_test)
print(new_train_set.shape)
print(new_test_set.shape)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(new_train_set, y_train)
print(
 "classifier accuracy {:.2f}%".format(
 classifier.score(new_test_set, y_test) * 100
 )
)
email = [
 "hey Gauri can you get together for lunch",
 "upto 70% discount on exclusive offer",
]
new_email = vectorizer.transform(email)
classifier.predict(new_email)

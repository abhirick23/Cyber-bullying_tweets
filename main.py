# import libraries
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# reading dataset
df = pd.read_csv(r"E:\NLP_based\tweets\cyberbullying_tweets.csv")
df.head()

df['cyberbullying_type'].value_counts()

sns.countplot(x='cyberbullying_type',data=df)

#clean dataset
import re
def  clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    return df
df_clean = clean_text(df, "tweet_text")

xfeatures  = df_clean['tweet_text']
ylabels = df_clean['cyberbullying_type']

#spliting data
x_train,x_test,y_train,y_test = train_test_split(xfeatures,ylabels, test_size=0.3, random_state=42)

from sklearn.pipeline import Pipeline
pipe_lr= Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(x_train,y_train)
pipe_lr.score(x_test,y_test)


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)
print(train_vectors.shape, test_vectors.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_vectors, y_train)

clf.score(test_vectors,y_test)

ex = " finally left school, no more childhood bullies, starting a new page, couldve been a lot better"

pipe_lr.predict([ex])

pipe_lr.predict_proba([ex])
pipe_lr.classes_

# save model and pipeline
import joblib
pipeline_file =  open("emotion_classiication_pipe_lr.pkl","wb")
joblib.dump(pipe_lr,pipeline_file)
pipeline_file.close()
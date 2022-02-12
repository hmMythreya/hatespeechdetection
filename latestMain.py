#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


df = pd.DataFrame(pd.read_csv("40k_modified.csv"))


# In[3]:


df.columns = ["ind","text","label"]


# In[4]:


df = df.drop(columns=["ind"])


# In[5]:


import random
dft=df[0:0]
corpus1 = []
corpus=[]
for i in df["text"]:
    corpus1.append(i)
for i in range(5000):
    in1=random.randint(0,40000)
    corpus.append(corpus1[in1])
   # print(df.index)
    dft.loc[i]=df.loc[in1]
  
print(dft)


# In[6]:


df = dft
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords")


# In[7]:


filter = set(stopwords.words("english"))


# In[8]:


final_corpus = []
for i in corpus:
    app_str = ""
    for x in i.split(" "):
      if x.lower() not in filter:
        app_str = app_str + x + " "
    final_corpus.append(app_str.lower())
final_corpus[0]


# In[9]:


vectorizer = CountVectorizer(analyzer="word",ngram_range=(1,3))
X = vectorizer.fit_transform(final_corpus)


# In[10]:


xDense = X.todense()
xDenseList=xDense.tolist()


# In[11]:


features = vectorizer.get_feature_names()
df2 = pd.DataFrame(xDenseList,columns=features)


# In[12]:


df2["Label"] = list(df.label)


# In[13]:


x_train = df2.sample(frac=0.9,random_state=0)


# In[14]:


x_test = df2.drop(x_train.index)


# In[15]:


df_hate = df2[df2.Label==1]
df_nothate = df2[df2.Label==0]


# In[16]:


x_train = df_hate.sample(frac=0.9,random_state=0)
y_train = df_nothate.sample(frac=0.9,random_state=0)


# In[17]:


x_test = df_hate.drop(x_train.index)
y_test = df_nothate.drop(y_train.index)


# In[18]:


x_train_df = pd.concat([x_train,y_train],axis=0)


# In[19]:


x_train_text = x_train_df.drop(["Label"],axis=1)


# In[20]:


y_train_label = list(x_train_df.Label)


# In[21]:


x_test_df = pd.concat([x_test,y_test],axis=0)


# In[22]:


x_test_text = x_test_df.drop(["Label"],axis=1)
y_test_label = x_test_df.Label


# In[23]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train_text,y_train_label)


# In[24]:


predict=logreg.predict(x_test_text)
from sklearn.metrics import accuracy_score
accuracy_score(y_test_label,predict)


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer
input1=["dalits are worthless"]
vtz = CountVectorizer(analyzer="word",ngram_range=(1,3))
tvtz = vtz.fit_transform(input1)
newDf = pd.DataFrame(tvtz.todense().tolist(),columns=vtz.get_feature_names())
predDf = x_train_text[0:0]

li = []
for i in predDf.columns:
  if i in newDf.columns:
    li.append(newDf[i])
  else:
    li.append(0.0)

predDf.loc[0] = li
if(logreg.predict(predDf)[0]):
  print("TOXIC!")
else:
  print("No hate was detected")


# In[ ]:


predDf


# In[ ]:


import pickle 
with open("model_LR","wb") as f:
  pickle.dump(logreg,f)


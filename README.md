# hatespeechdetection
A machine learning model to detect hate speech in tweets in english, hindi, and hindi-english mixed.
from sklearn.feature_extraction.text import CountVectorizer
input1=['']
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

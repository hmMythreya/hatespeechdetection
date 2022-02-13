import tkinter as tk
import os
import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

root = tk.Tk()
root.minsize(1200,900)

mod = pickle.load(open("model_ens","rb"))
mod_hi = pickle.load(open("model_ens_hi","rb"))
temp = pickle.load(open("train","rb"))
temp_hi = pickle.load(open("train_hi","rb"))

def makepredict():
    input1 = textfield.get(1.0,"end-1c")
    vtz = CountVectorizer(analyzer="word",ngram_range=(1,3))
    tvtz = vtz.fit_transform([input1])
    newDf = pd.DataFrame(tvtz.todense().tolist(),columns=vtz.get_feature_names())
    predDf = temp[0:0]

    li = []
    for i in predDf.columns:
        if i in newDf.columns:
            li.append(newDf[i])
        else:
            li.append(0.0)

    predDf.loc[0] = li
    
    #print("IN HINDI")
    predDf_hi = temp_hi[0:0]
    newDf_hi = pd.DataFrame(tvtz.todense().tolist(),columns=vtz.get_feature_names())
    li_h = []
    for i in predDf_hi.columns:
        if i in newDf_hi.columns:
            li_h.append(newDf[i])
        else:
            li_h.append(0.0)

    predDf_hi.loc[0] = li_h

    print(mod.predict_proba(predDf)[0])
    print(mod_hi.predict_proba(predDf_hi)[0])
    if(max(mod.predict_proba(predDf)[0])>max(mod_hi.predict_proba(predDf_hi)[0])):
        print("English")
        if(mod.predict(predDf)[0]):
            lbl.config(text="Hate Speech Detected!")
            root.configure(background="red")
        else:
            lbl.config(text="No hate was detected")
            root.configure(background="green")
    else:
        print("Hindi")
        if(mod_hi.predict(predDf_hi)[0]):
            lbl.config(text="Hate Speech Detected!")
            root.configure(background="red")
        else:
            lbl.config(text="No hate was detected")
            root.configure(background="green")


textfield = tk.Text()
textfield.pack()
button = tk.Button(
    root,
    text="GO",
    command = makepredict
)
lbl = tk.Label(root,text="")
button.pack()
lbl.pack()

root.title("Hate Speech Detection")
root.mainloop()
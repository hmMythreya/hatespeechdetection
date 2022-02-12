import tkinter as tk
import os
import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

root = tk.Tk()
root.minsize(1200,900)

mod = pickle.load(open("model_ens","rb"))
temp = pickle.load(open("train","rb"))

def predict():
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
    if(mod.predict(predDf)[0]):
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
    command = predict
)
lbl = tk.Label(root,text="")
button.pack()
lbl.pack()

root.title("Hate Speech Detection")
root.mainloop()
import tkinter as tk
import os
import sys

root = tk.Tk()
root.minsize(1200,900)

def saveInput():
    inpText = textfield.get(1.0,"end-1c")
    file1 = open("INPUT.txt","w+")
    file1.write(inpText)
    file1.close()
    try:
        from . import mainModel
    except ImportError:
        lbl.config(text="Main file missing!")
#   offensive = model.predict(inpText)
    


textfield = tk.Text()
textfield.pack()
button = tk.Button(
    root,
    text="GO",
    command = saveInput
)
lbl = tk.Label(root,text="")
button.pack()
lbl.pack()


root.mainloop()
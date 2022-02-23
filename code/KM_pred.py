from tkinter import *
from pandas import read_pickle
from pandas import to_pickle
import tkinter.font as tkFont
from KM_prediction import *


def evaluate(event):
    df = KM_predicton(substrate_list = [entry2.get()], 
             enzyme_list = [entry1.get()])
    res.configure(text = "KM: " + str(df["KM [mM]"][0]) , font = fontStyle)
    
w = Tk()
fontStyle = tkFont.Font(family="Lucida Grande", size=15)
Label(w, text="Amino acid sequence:", width= 30, height = 4, font=fontStyle).pack()
entry1 = Entry(w, font = fontStyle)
entry1.bind("<Return>", evaluate)
entry1.pack()

fontStyle = tkFont.Font(family="Lucida Grande", size=15)
Label(w, text="InChI string:", width= 30, height = 4, font=fontStyle).pack()
entry2 = Entry(w, font = fontStyle)
entry2.bind("<Return>", evaluate)
entry2.pack()

res = Label(w)
res.pack()
w.mainloop()
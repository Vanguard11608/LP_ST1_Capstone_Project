import joblib
import pandas as pd
import numpy as np
from tkinter import *
import spacy.cli
from spacy.lang.en.stop_words import STOP_WORDS

stop = STOP_WORDS
##spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
reg = joblib.load('Symptom2DiseaeModel1')

root = Tk()
root.geometry("500x300")

label = Label(text="Enter Your Main Symptom:")
label.pack()
userinput = Entry(root, width=30)
userinput.pack()
newU = userinput
newUInput = "I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches."


doc = nlp(newUInput)



def preprocess(doc):
    hold = []
    for token in doc:
        if token.is_space or token.is_punct or token.is_stop:
            continue

        print(token.lemma_)
        myvector = nlp(token.lemma_).vector
        hold.append(myvector)
        print(myvector)

    return hold


def vector(listof):
    vectorlistof = np.array(listof)

    print(vectorlistof)
    return vectorlistof


def diff(inputvectored):
    xtest = inputvectored
    xtest2 = np.stack(xtest)
    preds = reg.predict(xtest2)

    print(preds)
    print("The disease is: ")
    print(preds[0])

    ##out = str(preds)

    ##result_label = Label(out)
    ##result_label.pack()


def disease_prediction():
    imp = preprocess(doc)
    what = vector(imp)
    results = diff(what)
    print(results)



button = Button(root, text="Convert", command=disease_prediction)
button.pack()

root.mainloop()

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open('C:\\musab\\SDP Project\\intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    
    sentence_words = nltk.word_tokenize(sentence)
    
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words



def bag_of_words(sentence, words, show_details=True):
     
    sentence_words = clean_up_sentence(sentence)
    
    bag = [0]*len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result







# **********************************************?

#Creating tkinter GUI
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="white", font=("Times", 12 ))
    
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        
        ChatBox.insert(END, "Bot: " + res + '\n\n')
            
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
 

root = Tk()
root.title("Anxiety medication")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatBox = Text(root, bd=5, bg="gray", height="9", width="55", font=("Times",12))

ChatBox.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="dotbox")
ChatBox['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(root, font=("Times",15,'bold'), text="Send", width="12", height=5,
                    bd=5, bg="cyan", activebackground= "cyan2",cursor="heart",
                    command= send )

#Create the box to enter message
EntryBox = Text(root, bd=5, bg="medium turquoise",width="29", height="5", font="Time")

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatBox.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()

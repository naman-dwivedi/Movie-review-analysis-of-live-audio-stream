#!/usr/bin/env python
# coding: utf-8

# ## Sentimental analysis of movie reviews:
# 
# Here for the sentimental analysis of movie reviews the approach is that i have trained a Naive Bayes classifier using nltk movie review corpus that is a part of nltk package itself and for live audio speech to text i have implemented google speech recognition library. 

# #### First we are creating out bag of words which will be used to extract bigrams and unigrams from input speech/text.

# In[17]:


from nltk import ngrams
from nltk.corpus import stopwords 
import string
 
stopwords_english = stopwords.words('english')
 
# clean words, i.e. remove stopwords and punctuation
def clean_words(words, stopwords_english):
    words_clean = []
    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)    
    return words_clean 
 
 

 

    


 
 


# #### Feature extraction function for unigrams

# In[18]:


# feature extractor function for unigram
def bag_of_words(words):    
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary


# #### Feature extraction function for ngrams

# In[21]:


def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)    
    return words_dictionary



# In[23]:


from nltk.tokenize import word_tokenize
text = "It was a not very good movie."
words = word_tokenize(text.lower())
 
print ("Unigrams", words)
 
print (bag_of_ngrams(words))
 
words_clean = clean_words(words, stopwords_english)
print (words_clean)
 
important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'not', 'most', 'such', 'no', 'nor', 'only', 'so', 'than', 'too', 'very', 'just', 'but']
 
stopwords_english_for_bigrams = set(stopwords_english) - set(important_words)
 
words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)


# In[3]:


# We will use general stopwords for unigrams 
# And special stopwords list for bigrams
unigram_features = bag_of_words(words_clean)
print (unigram_features)
 
bigram_features = bag_of_ngrams(words_clean_for_bigrams)
print (bigram_features)


# In[5]:


# combine both unigram and bigram features
all_features = unigram_features.copy()
all_features.update(bigram_features)
print ("All features are  as follows", all_features)
 
# let's define a new function that extracts all features
# i.e. that extracts both unigram and bigrams features
def bag_of_all_words(words, n=2):
    words_clean = clean_words(words, stopwords_english)
    words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)
 
    unigram_features = bag_of_words(words_clean_for_bigrams)
    bigram_features = bag_of_ngrams(words_clean_for_bigrams)
 
    all_features = unigram_features.copy()
    all_features.update(bigram_features)
 
    return all_features
 
print (bag_of_all_words(words))


#  #### Now lets prepare our dataset i.e. movie review corpus here for model training.

# In[6]:


from nltk.corpus import movie_reviews 
 
pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)
 
neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)


# #### Here i have implemented  bag of words to reviews provided by both sets pos and neg respectively.

# In[7]:


pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_all_words(words), 'Review is positive'))
 
# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_all_words(words), 'Review is negative'))


# #### Creating test and training sests.
# Here the data for testing and training is being divided and then shuffle is used for output accuracy to be different everytime.

# In[24]:


print (len(pos_reviews_set), len(neg_reviews_set)) # Output: (1000, 1000)
 

from random import shuffle 
shuffle(pos_reviews_set)
shuffle(neg_reviews_set)
 
test_set = pos_reviews_set[:200] + neg_reviews_set[:200]
train_set = pos_reviews_set[100:] + neg_reviews_set[100:]
 
print(len(test_set),  len(train_set))


# #### This part of code trains the classifier and then print the accuracy gained(which can be different evertime)
# 

# In[25]:


from nltk import classify
from nltk import NaiveBayesClassifier
 
classifier = NaiveBayesClassifier.train(train_set)
 
accuracy = classify.accuracy(classifier, test_set)
print(accuracy)


# #### Voice input
# Here we are first taking input in the formm of live audio stream and then converting it into text.

# In[26]:


import wave
import pyaudio
from os import path
from pydub import AudioSegment
import nltk
import nltk.corpus
import os
import re
from werkzeug.utils import secure_filename
from flask import send_from_directory
from werkzeug.utils import secure_filename
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
#from sklearn.feature_extraction.text import CountVectorizer   

            
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 15    #set time for audio input
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()

 
r = sr.Recognizer() #recognizer instance

# this will start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK)
print ("Please speak your review for the movie...")
frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print ("Finished recording converting input into text")
 
 
#this part stops recording
stream.stop_stream(     )
stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

src = "file.wav"
dst = "audio.wav"
sound = AudioSegment.from_mp3(src)
audio = sound.export(dst, format="wav")

#parsing the audio file in speech recognizer
with sr.AudioFile("audio.wav") as source:
    #audio_text = r.record(source)
    audio = r.adjust_for_ambient_noise(source)
    audio_in  = r.listen(source)
    str = r.recognize_google(audio_in)
    try:
        print("You said: " + str)   
    except LookupError:                                 
        print("Could not understand your review")
    
                        


# #### Review analysis.
# Review analysis is done in this part here we are passing our input audio in the form of string(str) and then applying out earlier defined function bag of words to this customised review so the classifier will judge the review as its output.

# In[29]:


from nltk.tokenize import word_tokenize
 
custom_review = str
print("You said:", custom_review)
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_all_words(custom_review_tokens)
print(custom_review_set)
output= classifier.classify(custom_review_set)
print(output)


# In[ ]:





# In[ ]:





# In[9]:





# In[20]:





# In[ ]:





# In[22]:





# In[ ]:





# In[ ]:





# In[ ]:





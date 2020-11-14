# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:28:08 2020

@author: shankar
"""

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    sentences = []
    file = open(file_name, 'r') 
    f_data = file.readlines()
    f_data = [x for x in f_data if x != '\n'] # it should remove any break present
    f_data = [x.replace('\n',' ') for x in f_data] #this would remove that end of line
    f_data = ''.join(f_data) 
    article = f_data.split('. ') 
    for sentence in article:
        sentences.append(sentence.replace("^[a-zA-Z0-9!@#$&()-`+,/\"]", " ").split(" "))
    return sentences


def sentence_similarity(s1,s2,stopwords=None):
    if stopwords is None:
        stopwords=[]
    s1 = [w.lower() for w in s1]
    s2 = [w.lower() for w in s2]
    all_words = list(set(s1+s2))
    
    vector1 =[0]*len(all_words)
    vector2 =[0]*len(all_words)
    
    for w in s1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    
    for w in s2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
        
    # now, finding a cosine similarity between 2 vectors
    
    cosine_similarity = cosine_distance(vector1, vector2)
    
    return 1 - cosine_similarity

def generate_similarity_matrix(sentences , stop_words):
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
                                 
    for id1 in range(len(sentences)):
        for id2 in range(len(sentences)):
            if id1 == id2:
                continue
            similarity_matrix[id1][id2] = sentence_similarity(sentences[id1], sentences[id2], stop_words)
    return similarity_matrix

def generate_summary(file_name,top_n=5):
    stop_words = stopwords.words('english')
    summerize_text = []
    sentences = read_article(file_name)
    print(sentences)
    sentence_similarity_matrix = generate_similarity_matrix(sentences, stop_words)
    
    # need to rank sentences in similarity matrix
    
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    
    scores = nx.pagerank(sentence_similarity_graph)
    
    ranked_sentence = sorted(((scores[i],s)for i,s in enumerate(sentences)),reverse=True)
    print(ranked_sentence)
    for i in range(top_n):
        summerize_text.append(" ".join(ranked_sentence[i][1]))
    print("summary \n",". ".join(summerize_text))
    
generate_summary("random_para.txt", 2)
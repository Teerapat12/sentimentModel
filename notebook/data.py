import pickle
import pandas as pd
import numpy as np
from collections import Counter

def load_tfidf():
    X = pickle.load(open("../pickle/X_tfidf.pickle","rb"))
    y = pickle.load(open("../pickle/y_tfidf.pickle","rb"))
    return (X,y)

def load_bow():
    X = pickle.load(open("../pickle/X_bow.pickle","rb"))
    y = pickle.load(open("../pickle/y_bow.pickle","rb"))
    return (X,y)


def getScoredAmt(row):
    row = row.fillna("None")
    if(row['score_0']=="None"): return 0
    elif(row['score_1']=="None"): return 1   
    elif(row['score_2']=="None"): return 2
    else: return 3

def update_data():
    dfToken = pd.read_csv("../dataset/tokenized_comment_all_2.tsv", error_bad_lines=False,encoding='utf-8',sep='\t')
    dfScore = pd.read_csv("../dataset/facebook_comment_annotated_200717.tsv", error_bad_lines=False,encoding='utf-8',sep='\t')
    dfScore = dfScore[pd.notnull(dfScore['message'])]
    dfScore.reset_index(inplace=True)
    tokenizedCol = dfToken['tokenized']
    dfScore['token'] = tokenizedCol
    dfScore['service'][dfScore['service']==True] = 'TRUE'
    dfScore['scoredAmt'] = dfScore.apply(getScoredAmt,axis=1)
    dfScore.to_csv('../dataset/facebook_comment_tokenized_scored.tsv',sep='\t',encoding='utf-8',index=False)
    
    df = pd.read_csv("../dataset/facebook_comment_tokenized_scored.tsv",sep='\t',encoding='utf-8')
    df['token_arr'] = df['token'].apply(lambda x:eval(x))

    def giveClass(score):
        if(score==-32 ):
            return -32
        if(score==0):return 0
        if(score>0):return 1
        if(score<0):return -1
        
    scored = df[df['scoredAmt']>0]
    scored['class'] = scored['score'].apply(giveClass)
    
    comments = [i for i in scored['token_arr'].values]
    vocabs = Counter([word for comment in comments for word in comment])
    comments = [" ".join(i) for i in scored['token_arr'].values]

    #Creating tf-idf
    print("Creating the tf-idf matrix...\n")
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    tfidf_vectorizer = TfidfVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = [" "],   \
                                 max_features = 15000) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    tfidf_train_features = tfidf_vectorizer.fit_transform(comments)
    tfidf_train_features = tfidf_train_features.toarray()
    print("tf-idf features created")
    pickle.dump(tfidf_train_features,open("../pickle/X_tfidf.pickle","wb"))
    pickle.dump(scored['class'],open("../pickle/y_tfidf.pickle","wb"))
    
    print("Creating the bag of words...\n")
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = [" "],   \
                                 max_features = 15000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(comments)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features = train_data_features.toarray()
    print("Done!")
    
    pickle.dump(train_data_features,open("../pickle/X_bow.pickle","wb"))
    pickle.dump(scored['class'],open("../pickle/y_bow.pickle","wb"))
    
    
    
    
    
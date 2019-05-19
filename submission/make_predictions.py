import json
import string
import pickle
import numpy as np
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



def outputData(filename, datas):
    with open(filename, 'w+') as f:
        for data in datas:
            f.write('{}\n'.format(data))


# In[6]:


filename = "final_model.pth"
trained_model = pickle.load(open(filename, 'rb'))
#trained_model = model
predictions = trained_model.predict(X)

outputData('predictions.txt', predictions)
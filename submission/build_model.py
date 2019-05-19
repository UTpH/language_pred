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


# ### Mutinomial Naive Bayes

# In[2]:


#getting the data in shape
def getData(file_path, key):
    strings = []
    with open(file_path, 'r') as f:
        for idx_line, line in enumerate(f.readlines()):
            strings.append(json.loads(line)[key])

    return strings


def getFinalModelParams(X, Y, model):

    text_pipeline = Pipeline([('vectorizer', CountVectorizer()),
                              ('tfidf_transformator', TfidfTransformer()),
                              ('mod', model),
                             ])

    pipeline_parameters = {
        'vectorizer__analyzer': ['char_wb', 'word'],
        'vectorizer__ngram_range': [(2, 5)],
        'tfidf_transformator__use_idf': [True],
        'mod__alpha': [1e-3],
    }

    grid_search_clf = GridSearchCV(text_pipeline, pipeline_parameters, verbose=5)
    return grid_search_clf.fit(X, Y)


# In[2]:


def saveModel(model):
    filename = "final_model.pth"
    pickle.dump(model, open(filename, 'wb'))
    print("Saved Model!")


# In[4]:


X = np.array(getData('train_X_languages_homework.json.txt', 'text'))
Y = np.array(getData('train_y_languages_homework.json.txt', 'classification'))

#NB for sanity check
model = MultinomialNB()

## Get params of best model
model_gridS = getFinalModelParams(X, Y, model)
print('GridSearch found best score as: {} with params {}'.format(model_gridS.best_score_, model_gridS.best_params_))

## Save model
final_model = model_gridS.best_estimator_
saveModel(final_model)

## Save result
open('performance.txt', 'w+').write('GridSearch found best score as: {} with params {}'.format(model_gridS.best_score_, model_gridS.best_params_))


# In[4]:


def getData(file_path, key):
    strings = []
    with open(file_path, 'r') as f:
        for idx_line, line in enumerate(f.readlines()):
            strings.append(json.loads(line)[key])

    return strings

def getFinalModelParams(X, Y, model):

    text_pipeline = Pipeline([('vectorizer', CountVectorizer()),
                              ('tfidf_transformator', TfidfTransformer()),
                              ('mod', model),
                             ])

    pipeline_parameters = {
        'vectorizer__analyzer': ['char_wb', 'word'],
        'vectorizer__ngram_range': [(2, 5)],
        'tfidf_transformator__use_idf': [True],
    }

    grid_search_clf = GridSearchCV(text_pipeline, pipeline_parameters, verbose=5)
    return grid_search_clf.fit(X, Y)
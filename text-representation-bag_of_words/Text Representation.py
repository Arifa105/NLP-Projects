#!/usr/bin/env python
# coding: utf-8

# # Text Representation

# In[57]:


import pandas as pd
import numpy as np


# In[58]:


df = pd.read_csv("spam.csv")
df.head()


# # Drop column that are not relevent for the model 

# In[59]:


df1 = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis ='columns')


# In[60]:


df1.head()


# In[61]:


df1.Category.value_counts()


# In[ ]:


"""def get_spam_number(x):
    if x == 'spam':
        return 1
    return 0"""
# you can add the saame fuction in the apply function given below or you can use lambda function, both will work same


# # Create a new column

# In[62]:


#add transformation function
df1['spam'] = df1.Category.apply(lambda x: 1 if x =='spam' else 0)


# In[63]:


df1.head()


# In[64]:


df1.shape


# # split data into training and test

# In[65]:


# 20 for test and 80% for training data set (0.2 means 20%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df1.Message, df1.spam, test_size=0.2)


# In[66]:


X_train.shape


# In[ ]:


# above command show that 20% dataset deducted for the test so it shows total 4457 for training dadaset


# In[67]:


X_test.shape


# In[68]:


type(X_train)


# In[69]:


X_train[:4]


# In[70]:


type(y_train)


# In[71]:


y_train[:4]


# In[72]:


type(X_train.values)


# # Create bag of words representation using CountVectorizer

# In[73]:


from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer()

X_train_cv = v.fit_transform(X_train.values)
X_train_cv


# In[74]:


#covert multidimensional (2d array)
X_train_cv.toarray()[:2][0]


# In[75]:


X_train_cv.shape


# In[76]:


# v means found vectorizer
v.get_feature_names_out()[1000:1050]


# In[77]:


v.get_feature_names_out().shape


# In[ ]:


#dir is a method can tell you all the method which are supported on the varibales
dir(v)


# In[78]:


#give the vocabulary index position
v.vocabulary_


# In[79]:


X_train_np = X_train_cv.toarray()
X_train_np[:4]


# In[80]:


np.where(X_train_np[0]!=0)


# In[95]:


X_train[:4][300]


# In[91]:


X_train_np[0][6753]


# # Train the naive bayes model

# In[97]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_cv, y_train)


# In[1]:


X_test_cv = v.transform(X_test)


# # Evaluate Performance

# In[99]:


from sklearn.metrics import classification_report
y_pred = model.predict(X_test_cv)

print(classification_report(y_test, y_pred))


# In[100]:


emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

emails_count = v.transform(emails)
model.predict(emails_count)


# # Train the model using sklearn pipeline and reduce number of lines of code

# In[101]:


from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])


# In[102]:


clf.fit(X_train, y_train)


# In[103]:


y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





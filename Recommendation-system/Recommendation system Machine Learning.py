#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


credits.head()


# # marge both data set

# In[5]:


movies.merge(credits, on='title')


# In[6]:


movies.merge(credits, on='title').shape


# In[7]:


movies.shape


# In[8]:


credits.shape


# In[9]:


movies = movies.merge(credits, on='title')


# In[10]:


movies.head(1)


# # important column for analysis
# 1. genre
# 2. id
# 3. keywords
# 4. title
# 5. overview
# 6. cast
# 7. crew

# In[11]:


movies['original_language'].value_counts()


# In[12]:


movies.info()


# In[13]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[14]:


movies.head()


# # Preprocessing

# In[15]:


#Check the missing value
movies.isnull().sum()


# In[17]:


# Three(3) movies that do not have overview so we drop them
movies.dropna(inplace=True)


# In[18]:


#check the duplicate data
movies.duplicated().sum()


# In[19]:


movies.iloc[0].genres


# In[20]:


#string given error so we have to convert string of list into list so we import ast library
import ast


# In[21]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name']) 
    return L 


# In[22]:


movies['genres']= movies['genres'].apply(convert)


# In[23]:


movies.head()


# In[24]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[25]:


movies.head()


# In[26]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name']) 
            counter+= 1
        else:
            break
    return L 


# In[27]:


movies['cast'] = movies['cast'].apply(convert3)


# In[28]:


movies.head()


# In[29]:


movies['crew'][0]


# In[30]:


#dataset column data is in dictionar format so we want to extract job key from crew column
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[31]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[32]:


movies.head()


# In[33]:


movies['overview'][0]


# In[34]:


#overview column is in string so we convert it into list
movies['overview'].apply(lambda x: x.split())


# In[35]:


movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies.head()


# # Apply Transformation

# In[36]:


'''remove the space between the words like sam worthington so this one full name
but it count as a seperate tag so we have to remove the sapce between the full name like samworthington'''

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[37]:


movies.head()


# In[50]:


#create a new column and concatenate the all column in it
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[51]:


movies.head()


# In[52]:


# crete a new data frame with movie id ,title and tags 
new_df = movies[['movie_id', 'title', 'tags']] 


# In[56]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[57]:


new_df.head()


# In[71]:


#import after the vectors_features_names
import nltk


# In[72]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[77]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[79]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[58]:


new_df['tags'][0]


# In[59]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[60]:


new_df.head()


# In[61]:


new_df['tags'][0]


# In[62]:


new_df['tags'][1]


# # Feature Extraction

# In[80]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[81]:


# cv.fit_transform(new_df['tags']).toarray()
#cv.fit_transform(new_df['tags']).toarray().shape
vectors = cv.fit_transform(new_df['tags']).toarray()


# In[82]:


vectors


# In[83]:


vectors[0]


# In[84]:


cv.get_feature_names()


# In[74]:


['loved', 'love', 'loving']


# In[76]:


ps.stem('loving')


# In[85]:


from sklearn.metrics.pairwise import cosine_similarity


# In[87]:


#check the similiarity score of first with all
#other movies like haar movie ka haar ke sath distance find out krna hai similarity nikalni hai , 1 means similiarity ziada hai
#0 means similarity kaam hai
similarity = cosine_similarity(vectors)


# In[88]:


similarity


# In[89]:


new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[ ]:


# first jab movie find out krne ke liye firts have to find out index,
# index = new_df[new_df['title'] == 'batman begins'].index[0]
#distance nikalne hain
#then sort distances=> distances was in array
#sorted(similarity[0])
#distances = sorted(similarity[index])


# In[90]:


movie_index = new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[91]:


movie_index


# In[94]:


sorted(similarity[0][-10:-1])


# In[98]:


#list(enumerate(similarity[0]))
#sorted(list(enumerate(similarity[0])),reverse=True ,key = lambda x: x[1])
sorted(list(enumerate(similarity[0])),reverse=True ,key = lambda x: x[1])[1:6]


# In[101]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True ,key = lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[105]:


recommend('Avatar')


# In[103]:


new_df.iloc[1216].title


# In[ ]:





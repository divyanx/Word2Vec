#!/usr/bin/env python
# coding: utf-8

# In[15]:


corpus = ['king is a strong man', 
          'queen is a wise woman', 
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong', 
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']


# In[29]:


import json
import string
import re
import gdown
corpus = []
# read json file
fileName = 'Electronics_5.json'

# read
with open(fileName, "r") as read_file:
    i = 0
    l = read_file.readline()
    while l and i < 1000:
        i+=1
        l_dict = json.loads(l)
        review = l_dict["reviewText"]
        r_list = review.split('.')
        for r in r_list:
            r = re.sub('[^a-zA-Z]', ' ', r)
            r = r.lower()
            r = r.split()
            r = [w for w in r if not w in set(string.punctuation)]
            r = ' '.join(r)
            if r != '':
                corpus.append(r)
        l = read_file.readline()


# In[ ]:





# In[17]:


print(corpus[:10])


# In[18]:


from nltk.corpus import stopwords
def remove_stop_words(corpus):
    stop_words = set(stopwords.words('english'))
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    
    return results


# In[19]:


corpus = remove_stop_words(corpus)


# In[20]:


words = []
for text in corpus:
    for word in text.split(' '):
        #print(word)
        words.append(word)

print(len(words))
words = set(words)


# here we have word set by which we will have word vector

# In[21]:


print(words)


# In[22]:


word2int = {}

for i,word in enumerate(words):
    word2int[word] = i

sentences = []
for sentence in corpus:
    sentences.append(sentence.split())
    
WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] : 
            if neighbor != word:
                data.append([word, neighbor])


# In[23]:


import pandas as pd
for text in corpus:
    print(text)

df = pd.DataFrame(data, columns = ['input', 'label'])


# In[24]:


df.head(10)


# In[25]:


df.shape


# In[26]:


len(word2int)


# In[27]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np

ONE_HOT_DIM = len(words)


def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = [] 
Y = [] 
for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))


X_train = np.asarray(X)
Y_train = np.asarray(Y)


x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))


EMBEDDING_DIM = 50


W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1])) #bias
hidden_layer = tf.add(tf.matmul(x,W1), b1)


W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))


loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))


train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


# In[28]:


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) 

iteration = 20000
for i in range(iteration):

    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))


# In[ ]:



vectors = sess.run(W1 + b1)
print(vectors)


# In[ ]:


w2v_df = pd.DataFrame(vectors, columns = ['x'+str(i) for i in range(1, EMBEDDING_DIM+1)])
w2v_df['word'] = words
w2v_df = w2v_df[['word'] + ['x'+str(i) for i in range(1, EMBEDDING_DIM+1)]]
w2v_df


# In[ ]:


# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()

# for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
#     ax.annotate(word, (x1,x2 ))
    
# PADDING = 1.0
# x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
# y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
# x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
# y_axis_max = np.amax(vectors, axis=0)[1] + PADDING
 
# plt.xlim(x_axis_min,x_axis_max)
# plt.ylim(y_axis_min,y_axis_max)
# plt.rcParams["figure.figsize"] = (10,10)

# plt.show()


# In[ ]:


# save model to csv
w2v_df.to_csv('w2v_model.csv', index=False)


# In[ ]:


def cosine_similarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# In[ ]:


def findSimilarity(word, model, n = 10):
    # get the index of the word
  
    word_vec = list(w2v_df[w2v_df['word'] == word].values[0])[1:]
    similarities = []
    for index, row in w2v_df.iterrows():
        if row['word'] == word:
            continue
        similarities.append((row['word'], cosine_similarity(word_vec, list(row)[1:])))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:n]
    


# In[ ]:


print(findSimilarity('bigger', w2v_df, n=10))


# In[ ]:


# find the row where word column is 'love'


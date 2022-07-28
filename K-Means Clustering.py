#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[39]:


dataset = pd.read_csv('Mall_Customers.csv')


# In[40]:


dataset


# In[41]:


X = dataset.iloc[:,3:].values


# In[42]:


X


# In[58]:


from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=101)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[59]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=101)
kmeans.fit(X)
predictions = kmeans.predict(X)


# In[60]:


predictions


# In[72]:


plt.scatter(X[predictions == 0,0], X[predictions == 0,1], s=100, c='red', label = 'Cluster 1')
plt.scatter(X[predictions == 1,0], X[predictions == 1,1], s=100, c='blue', label = 'Cluster 2')
plt.scatter(X[predictions == 2,0], X[predictions == 2,1], s=100, c='green', label = 'Cluster 3')
plt.scatter(X[predictions == 3,0], X[predictions == 3,1], s=100, c='cyan', label = 'Cluster 4')
plt.scatter(X[predictions == 4,0], X[predictions == 4,1], s=100, c='magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c='yellow', label='Centroid')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()


# In[ ]:





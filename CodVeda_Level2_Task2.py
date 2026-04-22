#!/usr/bin/env python
# coding: utf-8

# # Iris Classification using Decision Tree 
# 
# ## Problem Statement
# Build a machine learning model that can classify iris flowers into different species based on their physical measurements.
# 
# ---
# 
# ## Goals
# - Construct a Decision Tree classifier
# - Interpret model decisions visually
# - Control overfitting using pruning
# - Measure performance using evaluation metrics
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[14]:


data = pd.read_csv("iris.csv")

data.columns = [col.lower().replace(" ", "_") for col in data.columns]

data.sample(5)


# In[3]:


print("Shape:", data.shape)
print("\nColumns:", data.columns.tolist())

data.describe()


# ## Class Distribution

# In[4]:


data['species'].value_counts().plot(kind='bar')
plt.title("Distribution of Target Classes")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()


# ## Feature Selection

# In[5]:


features = data.iloc[:, :-1]
target = data.iloc[:, -1]


# ## Data Splitting

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, stratify=target, random_state=1
)


# ## Building Decision Tree Model

# In[7]:


model = DecisionTreeClassifier(
    max_depth=3,   
    random_state=1
)

model.fit(X_train, y_train)


# ## Prediction

# In[8]:


predictions = model.predict(X_test)


# ## Model Performance

# In[9]:


print(classification_report(y_test, predictions))


# ## Confusion Matrix

# In[10]:


cm = confusion_matrix(y_test, predictions)

plt.imshow(cm)
plt.title("Confusion Matrix Visualization")
plt.colorbar()

plt.xticks(range(3), target.unique())
plt.yticks(range(3), target.unique())

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ## Tree Visualization

# In[11]:


from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=features.columns)
plt.title("Decision Tree Structure")
plt.show()


# ## Feature Contribution Analysis

# In[12]:


importance = model.feature_importances_

for f, val in zip(features.columns, importance):
    print(f"{f} → {round(val, 3)}")


# ## Accuracy Check

# In[13]:


from sklearn.metrics import accuracy_score

print("Model Accuracy:", accuracy_score(y_test, predictions))


# ## Conclusion
# 
# - Decision Trees provide intuitive and interpretable models
# - Pre-pruning helps control complexity
# - Model performs well on structured datasets like Iris
# 
# ---
# 
# ## Key Highlight
# 
# This implementation uses:
# 
# 1.Stratified splitting  
# 2.Pre-pruning instead of post-pruning  
# 3.Clean visualization  
# 4.Interpretability-focused approach  
# 

# In[ ]:





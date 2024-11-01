#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')


# In[11]:


import numpy as np


# In[15]:


import pandas as pd

# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Display the first few rows of the dataframe
df.head()


# In[19]:


missing_gender = df['gender'].isnull().sum()
print(missing_gender)


# In[35]:


df['diabetes'].isnull().sum()


# In[39]:


df['gender'].value_counts()


# In[41]:


df['gender'].unique()


# In[43]:


# Apply one-hot encoding to the 'gender' column
df_new = pd.get_dummies(df, columns=['gender'], prefix='gender')

# Display the first few rows of the encoded dataset
df_new.head()


# In[47]:


# Ensure the encoded columns are integers (0 and 1)
df_new[['gender_Female', 'gender_Male', 'gender_Other']] = df_new[['gender_Female', 'gender_Male', 'gender_Other']].astype(int)

# Display the first few rows to confirm
df_new.head()


# In[49]:


df['smoking_history'].unique()


# In[51]:


data = df_new[df_new['smoking_history'] != 'No Info']



data[data['smoking_history'] == 'never']


# In[65]:


data['smoking_history'].value_counts().get('current', 0)


# In[67]:


data_update = pd.get_dummies(data, columns=['smoking_history'])

# Display the first few rows to verify
print(data_update.head())



data_update[['smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 
             'smoking_history_never', 'smoking_history_not current']] = data_update[['smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 
                 'smoking_history_never', 'smoking_history_not current']].astype(int)

# Display the first few rows to verify
print(data_update.head())


# In[79]:


data_update.head()


# In[81]:


data_update.to_csv('data_state.csv', index=False)


# In[83]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Select numerical columns
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']



# In[87]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load your dataset
df = pd.read_csv('data_state.csv')

# Select numerical columns
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Create new DataFrame to store transformed data
transformed_df = df[numerical_cols].copy()

# 1. Min-Max Scaling
min_max_scaler = MinMaxScaler()
transformed_df[numerical_cols] = min_max_scaler.fit_transform(df[numerical_cols])

# 2. Standard Scaling (Z-score normalization)
standard_scaler = StandardScaler()
standard_scaled_df = pd.DataFrame(standard_scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

# 3. Log Transformation (handling zero or negative values by adding a constant)
log_transformed_df = df[numerical_cols].apply(lambda x: np.log1p(x))  # log1p is log(1 + x)

# Plot the transformed data distributions
plt.figure(figsize=(12, 12))

# Min-Max Scaled Data
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 3, i)
    sns.histplot(transformed_df[col], kde=True, bins=30, color='lightcoral')
    plt.title(f'Min-Max Scaled {col}')
    plt.xlabel(col)

# Standard Scaled Data
for i, col in enumerate(numerical_cols, 4):
    plt.subplot(4, 3, i)
    sns.histplot(standard_scaled_df[col], kde=True, bins=30, color='mediumseagreen')
    plt.title(f'Standard Scaled {col}')
    plt.xlabel(col)

# Log Transformed Data
for i, col in enumerate(numerical_cols, 7):
    plt.subplot(4, 3, i)
    sns.histplot(log_transformed_df[col], kde=True, bins=30, color='deepskyblue')
    plt.title(f'Log Transformed {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()



from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your dataset
df = pd.read_csv('data_state.csv')

# Select numerical columns
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Standard Scaling (Z-score normalization)
standard_scaler = StandardScaler()
df[numerical_cols] = standard_scaler.fit_transform(df[numerical_cols])

# Verify the standardization
df.describe()

# Create interaction features
df['hypertension_heart_disease'] = df['hypertension'] * df['heart_disease']
df['diabetes_age'] = df['diabetes'] * df['age']

# Display the first few rows to verify
df.head()


df.drop(['age_group', 'bmi_category'], axis=1, inplace=True) # Display the first few rows to verify df.head()



# Export dataframe to CSV
df.to_csv('data_model.csv', index=False)


# In[ ]:





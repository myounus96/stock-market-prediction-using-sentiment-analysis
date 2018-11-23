
# coding: utf-8

# In[265]:


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score


# In[266]:


# from google.colab import files

# uploaded = files.upload()

# for fn in uploaded.keys():
#   print('User uploaded file "{name}" with length {length} bytes'.format(
#       name=fn, length=len(uploaded[fn])))


# In[267]:


df=pd.read_csv('final_data.csv')


# In[268]:


df.head()


# In[269]:


X=df[['compound','neg','neu','pos']]
# X=df[['neg','neu','pos']]
# Y=df[['Close']]/400     #normalize it
Y=df[['Close']]


# In[270]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[271]:


from sklearn import preprocessing
min_max_scalar=preprocessing.MinMaxScaler()
X_train=min_max_scalar.fit_transform(X_train)
X_test=min_max_scalar.fit_transform(X_test)
Y_train=min_max_scalar.fit_transform(Y_train)
Y_test=min_max_scalar.fit_transform(Y_test)


# In[272]:


# from sklearn.preprocessing import StandardScaler
# scalar=StandardScaler()
# scalar.fit(X_train)
# X_train=scalar.transform(X_train)
# X_test=scalar.transform(X_test)


# In[273]:


from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import activations


# In[274]:


model=Sequential()


# In[275]:


model.add(Dense(4,activation=activations.sigmoid,input_shape=(4,)))
# model.add(Dense(3,activation='relu',input_shape=(3,)))
model.add(Dense(3,activation=activations.sigmoid))
model.add(Dense(2,activation=activations.sigmoid))
# model.add(Dense(100,activation=activations.sigmoid))
# model.add(Dense(100,activation=activations.sigmoid))
# model.add(Dense(100,activation=activations.sigmoid))
# model.add(Dense(100,activation=activations.sigmoid))
model.add(Dense(1,activation=activations.sigmoid))


# In[276]:


model.compile(optimizer='adam',loss=losses.mean_absolute_error)


# In[277]:


# while 1:
#     model.fit(X_train,Y_train,epochs=500)
#     y_pred=model.predict(X_test)
#     falto=r2_score(Y_test,y_pred)
#     print(falto)
#     if falto >= 70.0:
#         break


# In[278]:


model.fit(X_train,Y_train,verbose=2,epochs=1000000)
# model.fit(X_train,Y_train,verbose=2)


# In[243]:


y_pred=model.predict(X_test)


# In[244]:


model.evaluate(X_train,Y_train)


# In[245]:


model.evaluate(X_test,Y_test)


# In[246]:


print(r2_score(Y_test,y_pred))


# In[247]:


pd.DataFrame(model.predict(X)).to_csv('ansr_y_pred.csv')


# In[248]:


y_pred


# In[249]:


Y_test


# In[250]:


model.layers[0].get_weights()


# In[251]:


model.layers[1].get_weights()


# In[252]:


model.layers[2].get_weights()


# In[253]:


model.layers[3].get_weights()


# In[254]:


# model.save_weights("model.h5")


# In[255]:


model_json = model.to_json()


# In[256]:


model_json


# In[257]:


from keras.models import load_model


# In[258]:


# model.load_weights('model_full.h5')


# In[259]:


# model.layers[3].get_weights()


# In[260]:


# y_pred=model.predict(X_test)


# In[261]:


# print(r2_score(Y_test,y_pred))


# In[262]:


model.save('model_full.h5')


# In[263]:


# model=load_model('model_full.h5')


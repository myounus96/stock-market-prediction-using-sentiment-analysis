
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random, math


# In[2]:


random.seed(786)
np.random.seed(786)


# In[3]:


df = pd.read_csv('final_data.csv')
# df1 = df[['compound','neg','neu','pos','Close']]
df1 = df[['compound','neg','neu','pos']]
df2=df[['Close']]
# df = df[['neg','neu','pos','Close']]


# In[4]:


from sklearn import preprocessing
min_max_scalar=preprocessing.MinMaxScaler()
data=min_max_scalar.fit_transform(df1)
# X_test=min_max_scalar.fit_transform(X_test)
# Y_train=min_max_scalar.fit_transform(Y_train)
# Y_test=min_max_scalar.fit_transform(Y_test)


# In[5]:


df1=pd.DataFrame(data)


# In[6]:


df1.head()


# In[7]:


close_max=df2.max()
df2/=close_max
# df2=np.array(df2)


# In[8]:


df2.head()


# In[9]:


df1['4']=df2


# In[10]:


df1.head()


# In[11]:


# df['Close']=df['Close']/400


# In[12]:


data = df1.values.tolist()


# In[13]:


data[:2]


# In[14]:


# data = data.tolist()


# In[15]:


inner_threshold = 0.005;
rate_var = 0;

final_ansr = []


# In[16]:


sigmoid=lambda x:1/(1+math.e**(-x))


# In[17]:


def fit_func(rec, i):
    x1, x2, x3, x4, y = i[0], i[1], i[2], i[3], i[4]

    out5 = (x1 * rec['data']['w15']) + (x2 * rec['data']['w25']) + (x3 * rec['data']['w35']) + (x4 * rec['data']['w45']) + rec['data']['o5']
    out5 = sigmoid(out5)

    out6 = (x1 * rec['data']['w16']) + (x2 * rec['data']['w26']) + (x3 * rec['data']['w36']) + (x4 * rec['data']['w46']) + rec['data']['o6']
    out6 = sigmoid(out6)
    
    out7 = (x1 * rec['data']['w17']) + (x2 * rec['data']['w27']) + (x3 * rec['data']['w37']) + (x4 * rec['data']['w47']) + rec['data']['o7']
    out7 = sigmoid(out7)

    out8 = (out5 * rec['data']['w58']) + (out6 * rec['data']['w68']) + (out7 * rec['data']['w78']) + rec['data']['o8']
    out8 = sigmoid(out8)
    
    out9 = (out5 * rec['data']['w59']) + (out6 * rec['data']['w69']) + (out7 * rec['data']['w79']) + rec['data']['o9']
    out9 = sigmoid(out9)

    out10 = (out8 * rec['data']['w810']) + (out9 * rec['data']['w910']) + rec['data']['o10']
    out10 = sigmoid(out10)

    err10 = out10 * (1 - out10) * (y - out10)

    if abs(err10) <= inner_threshold:
        # final_ansr.append(out6 * 20000)
        rec['fitness'] += 1


# In[18]:


nList = [];
m = 20;
n = 25;
mutation_prob = 75;
mutation = 0.5;


# In[19]:


# ranges for weights & biases
wl = -2;
wr = 2;
bl = -1;
br = 1


# In[20]:


total_weights=20
total_biases=6


# In[21]:


for i in range(n):
    # floating random weights
    w15, w16, w17, w25, w26,w27, w35, w36,w37, w45, w46,w47, w58,w59,w68,w69,w78,w79, w810, w910 = np.random.uniform(wl, wr, total_weights)
    # floating random biases
    o5, o6, o7,o8, o9, o10 = np.random.uniform(bl, br, total_biases)

    nList.append({'data': {'w15': w15, 'w16': w16, 'w17': w17, 'w25': w25, 'w26': w26, 'w27': w27,
                           'w35': w35,'w36': w36, 'w37': w37, 'w45': w45, 'w46': w46,
                           'w47': w47, 'w58': w58, 'w59': w59, 'w68': w68, 'w69': w69, 'w78': w78,
                           'w79': w79,'w810': w810,'w910': w910, 'o5': o5, 'o6': o6, 'o7': o7,
                           'o8': o8, 'o9': o9, 'o10': o10},
                           'fitness': 0})


# In[22]:


# fitness calculation for first time/initial population
for rec in nList:
    for i in data:
        fit_func(rec, i)
        


# In[23]:


count = 0;
final_rec = 0
flag = False
stop_at=100000


# In[ ]:


while (1):
    count += 1
    mList = []

    print(f"gen: {count}")

    if count > stop_at:
        final_rec = sorted(nList, key=lambda f: f['fitness'], reverse=True)[0]
        break

    rec_num = 0
    for rec in nList:
        # print("rec_index: {}".format(rec_num))
        rec_num += 1
        if rec['fitness'] == len(data):
            print("rec_index: {}".format(rec_num))
            final_rec = rec
            # print('gen: {}'.format(count))
            print('Answer: {}'.format(rec))
            flag = True
            break

    if flag:
        break

    for j in range(int(m / 2)):
        # randomly parent selection(not same)
        # t=temp

        t = random.sample(range(n), 2)

        p1 = nList[t[0]]
        p2 = nList[t[1]]

        # crossover
        crossover_point=int((total_weights+total_biases)/2)
        ch1 = list(p1['data'].values())[:crossover_point] + list(p2['data'].values())[crossover_point:]
        ch2 = list(p2['data'].values())[:crossover_point] + list(p1['data'].values())[crossover_point:]

        # mutation work
        for k in [ch1, ch2]:
            if random.randint(0, 100) <= mutation_prob:
                # get random for mutation in weights
                t = random.sample(range(total_weights), 3)
                # get random for mutation in bias
                t1 = random.randint(total_weights, (total_weights+total_biases)-1)
                # odd,do +ve
                if random.randint(1, 2) == 1:
                    # mutation in weights
                    if (k[t[0]] + mutation) >= wl and (k[t[0]] + mutation) <= wr:
                        k[t[0]] += mutation

                    if (k[t[1]] + mutation) >= wl and (k[t[1]] + mutation) <= wr:
                        k[t[1]] += mutation

                    if (k[t[2]] + mutation) >= wl and (k[t[2]] + mutation) <= wr:
                        k[t[2]] += mutation

                    # mutation in bias
                    if (k[t1] + mutation) >= bl and (k[t1] + mutation) <= br:
                        k[t1] += mutation

                # even,do -ve
                else:
                    # mutation in weights
                    if (k[t[0]] - mutation) >= wl and (k[t[0]] - mutation) <= wr:
                        k[t[0]] -= mutation

                    if (k[t[1]] - mutation) >= wl and (k[t[1]] - mutation) <= wr:
                        k[t[1]] -= mutation

                    if (k[t[2]] - mutation) >= wl and (k[t[2]] - mutation) <= wr:
                        k[t[2]] -= mutation

                    # mutation in bias
                    if (k[t1] - mutation) >= bl and (k[t1] - mutation) <= br:
                        k[t1] -= mutation

                        
        ch1 = {'data': {'w15': ch1[0], 'w16': ch1[1], 'w17': ch1[2], 'w25': ch1[3], 'w26': ch1[4], 'w27': ch1[5],
                        'w35': ch1[6],
                        'w36': ch1[7], 'w37': ch1[8], 'w45': ch1[9], 'w46': ch1[10],
                        'w47': ch1[11], 'w58': ch1[12], 'w59': ch1[13], 'w68': ch1[14], 'w69': ch1[15], 'w78': ch1[16],
                        'w79': ch1[17],
                        'w810': ch1[18], 'w910': ch1[19], 'o5': ch1[20], 'o6': ch1[21],
                        'o7': ch1[22], 'o8': ch1[23], 'o9': ch1[24], 'o10': ch1[25]}, 'fitness': 0}

        ch2 = {'data': {'w15': ch2[0], 'w16': ch2[1], 'w17': ch2[2], 'w25': ch2[3], 'w26': ch2[4], 'w27': ch2[5],
                        'w35': ch2[6],
                        'w36': ch2[7], 'w37': ch2[8], 'w45': ch2[9], 'w46': ch2[10],
                        'w47': ch2[11], 'w58': ch2[12], 'w59': ch2[13], 'w68': ch2[14], 'w69': ch2[15], 'w78': ch2[16],
                        'w79': ch2[17],
                        'w810': ch2[18], 'w910': ch2[19], 'o5': ch2[20], 'o6': ch2[21],
                        'o7': ch2[22], 'o8': ch2[23], 'o9': ch2[24], 'o10': ch2[25]}, 'fitness': 0}


        # calculate fitness of children
        for rec in [ch1, ch2]:
            for i in data:
                fit_func(rec, i)

        mList.append(ch1)
        mList.append(ch2)

    # combine both mList and nList and sort it with respect fit func value,also ovverides the nList,so we used in next iteration

    nList = sorted(nList + mList, key=lambda item: item['fitness'], reverse=True)[:25]


# In[ ]:


for i in data:
    x1, x2, x3, x4, y = i[0], i[1], i[2], i[3], i[4]

    out5 = (x1 * final_rec['data']['w15']) + (x2 * final_rec['data']['w25']) + (x3 * final_rec['data']['w35']) + (x4 * final_rec['data']['w45']) + final_rec['data']['o5']
    out5 = sigmoid(out5)

    out6 = (x1 * final_rec['data']['w16']) + (x2 * final_rec['data']['w26']) + (x3 * final_rec['data']['w36']) + (x4 * final_rec['data']['w46']) + final_rec['data']['o6']
    out6 = sigmoid(out6)
    
    out7 = (x1 * final_rec['data']['w17']) + (x2 * final_rec['data']['w27']) + (x3 * final_rec['data']['w37']) + (x4 * final_rec['data']['w47']) + final_rec['data']['o7']
    out7 = sigmoid(out7)

    out8 = (out5 * final_rec['data']['w58']) + (out6 * final_rec['data']['w68']) + (out7 * final_rec['data']['w78']) + final_rec['data']['o8']
    out8 = sigmoid(out8)
    
    out9 = (out5 * final_rec['data']['w59']) + (out6 * final_rec['data']['w69']) + (out7 * final_rec['data']['w79']) + final_rec['data']['o9']
    out9 = sigmoid(out9)

    out10 = (out8 * final_rec['data']['w810']) + (out9 * final_rec['data']['w910']) + final_rec['data']['o10']
    out10 = sigmoid(out10)

    final_ansr.append(out10*close_max)


# In[ ]:


# final_ansr=final_ansr[-len(data):]
df['Y-ENN'] = pd.DataFrame(final_ansr)
df.to_csv('NN-DATA2.csv', index=False)


# In[ ]:


final_rec


# In[ ]:


from sklearn.metrics import r2_score
r2_score(df[['Close']],df['Y-ENN'])


# In[ ]:


df[['Close']].head()


# In[ ]:


df['Y-ENN'].head()


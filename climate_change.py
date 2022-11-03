#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

data = pd.read_csv("climate_change_updated.csv")
data.head(180)


# In[11]:


data_uzbek = data.rename(columns={"years":"yil", "month":"oy","maxC":"maks_harorat","MinC":"min_harorat"})


# In[4]:


Yuqori_harorat = data['maxC']
Past_harorat = data['minC']
plt.title("maxC vs minC")
plt.plot(Yuqori_harorat, color="DarkBlue", label="Yuqori harorat")
plt.plot(Past_harorat, color="DarkGreen", label="Past harorat")
plt.legend()
plt.show()


# In[27]:


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

names = ['maxC', 'minC']
values = [1, 10]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical diagram')
plt.show()


# In[3]:


import pandas as pd
from matplotlib import pyplot as plt
x = [1, 2, 3]
y = [0, 5, 10]
z = [10, 5, 0]
plt.plot(x,y)
plt.plot(x,z)
plt.title("Sodda chiziq")
plt.xlabel("x")
plt.ylabel("y and z")
plt.legend(["this is y", "this is z"])
plt.show()


# In[10]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

MaxC = ['maxC', 'precipitation']
counts = [15, 10]
maxtemps_labels = ['skyblue']
precip_labels = ['red']

ax.bar(maxC, precipitation, label=maxtemps_labels, color=(maxtemps_colors, precip_colors))

ax.set_ylabel('Degree of maxtemps', 'Quantity of precipitation')
ax.set_title('precipitation type')
ax.legend(title='precipitation color')

plt.show()


# In[9]:


# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the open interval (0, 1)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthresh=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()


# In[14]:


mu, sigma = 115, 15
x = mu + sigma * np.random.randn(10000)
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
# the histogram of the data
n, bins, patches = ax.hist(x, 50, density=True, facecolor='C0', alpha=0.75)

ax.set_xlabel('Volume')
ax.set_ylabel('maxC')
ax.set_title('Precipitation and maxC')
ax.text(75, .025, r'$\mu=115,\ \sigma=15$')
ax.axis([55, 175, 0, 0.03])
ax.grid(True);


# In[9]:


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

data = pd.read_csv("climate_change_updated.csv")
data.head(180)
x = np.linspace(0, 2)  # Sample data.

# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, ax = plt.subplots(figsize=(15, 10), layout='constrained')
ax.plot(x, x, label='minC')  # Plot some data on the axes.
ax.plot(x, x**2, label='maxC')  # Plot more data on the axes...
ax.plot(x, x**3, label='solar')  # ... and some more.
ax.set_xlabel('x label')  # Add an x-label to the axes.
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Range of maxC, minC and solar")  # Add a title to the axes.
ax.legend();  # Add a legend.


# In[26]:


fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(7, 2.7), layout='constrained')
l1, = ax1.plot(t, s)
ax2 = ax1.twinx()
l2, = ax2.plot(t, range(len(t)), 'C1')
ax2.legend([l1, l2], ['Sine (left)', 'Straight (right)'])

ax3.plot(t, s)
ax3.set_xlabel('Angle [rad]')
ax4 = ax3.secondary_xaxis('top', functions=(np.rad2deg, np.deg2rad))
ax4.set_xlabel('Angle [°]')


# In[ ]:


data_sorted = data.sort_values(by=['Month'], ascending=False)
cars_colors = ['tab:brown', 'tab:green', 'tab:brown', 'tab:blue']


# In[ ]:


eng_issiq_oylar = data_sorted.head(20)
eng_issiq_oylar


# In[ ]:


max_temp_oy = data.sort_values(by=['max','temp_month'],ascending=False)


# In[29]:


import matplotlib.pyplot as plt
import pandas as pd 

data = {
    "solar":[10, 5, 7],
    "humidity" : [-2, 6, 9],
    "year" : [2010, 2011, 2014]
}
df = pd.DataFrame(data)
  
plt.bar(X, Y, color='b')
plt.title("hot")
plt.xlabel("year")
plt.ylabel("month")
  
plt.show()


# In[ ]:


X = list(df.iloc[:, 0])
Y = list(df.iloc[:, 2])
  
plt.plot(X, Y, color='g')
plt.title("O'zbekiston aholisining bakterial dizenteriya bilan kasallanishi darajasi")
plt.xlabel("Yillar")
plt.ylabel("Bakterial dizenteriya, ming")
  
plt.show()


# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

issiq = pd.read_csv ('climate_change_updated.csv')
df = pd.DataFrame(issiq)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
X = list(df.iloc[:, 0])
Y1 = list(df.iloc[:, 3])
Y2 = list(df.iloc[:, 4])

kasallanish.plot()

plt.show()


# In[35]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dependence = pd.read_csv ('climate_change_updated.csv')
dependence
df = pd.DataFrame(dependence)
X = list(df.iloc[:, 0])
Y = list(df.iloc[:, 3])
  
plt.step(X, Y, color='b')
plt.title("temp_degree")
plt.xlabel("year")
plt.ylabel("precip, mm")
  
plt.show()


# In[52]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

df = pd.read_csv('climate_change_updated.csv')

fig, ax = plt.subplots(figsize=(15,4), nrows=1, ncols=3);
fig, ax = plt.subplots(figsize=(15,4), nrows=1, ncols=3);
ax[0].hist(df['years'])
ax[0].set_title('hot_year')
ax[1].hist(df['month'])
ax[1].set_title('hot_month')


# In[ ]:





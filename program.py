import pandas as pd
import numpy as np
import datetime
import scipy
import scikit_posthocs as sp
from datetime import date
from scipy import stats

in_directory = 'googleplaystore.csv'

def to_timestamp(x):
    return x.timestamp()

data = pd.read_csv(in_directory)
data = data.dropna()

data['Price'] = data['Price'].str.extract(r'(\d+.\d+)')[0].astype('float32')
data['Price'] = data['Price'].fillna(0)
data['Installs'] = data['Installs'].str.replace(r'[\+,]','').astype('int64')
data = data[data['Installs']>1000]

#parse the 'Last Updated' column to datetime type
data['Last Updated'] = pd.to_datetime(data['Last Updated'], format='%B %d, %Y')	

paid = data[data['Type'] == "Paid"]		# table of paid apps only

# convert datetime column to int timestamp so we can plot it
# timestamp = data['Last Updated'].apply(to_timestamp)
paid_timestamp = paid['Last Updated'].apply(to_timestamp)

#LINEAR REGRESSION on price of apps compared to the last update time
#Do more recent/ recently updated apps cost more?
time_price_fit = stats.linregress(paid_timestamp,paid['Price'])
# time_rating_fit = stats.linregress(paid_timestamp,paid['Rating'])

price_predict = paid_timestamp*time_price_fit.slope + time_price_fit.intercept

plt.plot(paid['Last Updated'], paid['Price'], 'b.', alpha=0.5)
plt.plot(paid['Last Updated'], price_predict, 'r-', linewidth=3)
plt.xlabel('Last update')
plt.ylabel('Price')
plt.title('Change in Price over time')
plt.savefig('Time vs Price')

# may or may not be useful?

	# rating_predict = paid_timestamp*time_rating_fit.slope + time_rating_fit.intercept
	# plt.plot(paid['Last Updated'], paid['Rating'], 'b.', alpha=0.5)
	# plt.plot(paid['Last Updated'], rating_predict, 'r-', linewidth=3)
	# plt.xlabel('Last update')
	# plt.ylabel('Price')
	# plt.title('Change in Rating over time')

category = data[['Category','App','Installs']].copy()
rank = category.groupby(['Category']).agg('count').sort_values('App', ascending = False)
# group = data.groupby(['Category','Installs']).agg('count')

# --getting the top 5 --
rankLst = rank.index.values.tolist()

rank1 = category[category['Category'] == rankLst[0]].reset_index()
rank2 = category[category['Category'] == rankLst[1]].reset_index()
rank3 = category[category['Category'] == rankLst[2]].reset_index()
rank4 = category[category['Category'] == rankLst[3]].reset_index()
rank5 = category[category['Category'] == rankLst[4]].reset_index()

var = stats.levene(rank1['Installs'],rank2['Installs']).pvalue
print('Levene test for equal variance p-value:', var)

# -- doing an kruskal test, similar to anova but for unequal var and sample size
Kruskal = stats.kruskal(rank1['Installs'], rank2['Installs'], rank3['Installs'], rank4['Installs'], rank5['Installs'])
print(Kruskal)
# print(Kruskal.pvalue)

# -- Doing a posthoc tamhane test, similar to tukey but for unequal var and sample size --
x = pd.concat([rank1['Installs'],rank2['Installs'],rank3['Installs'],rank4['Installs'],rank5['Installs']], ignore_index=True, axis=1)
x = x.melt(var_name='groups', value_name='values')
posthoc = sp.posthoc_tamhane(x, val_col='values', group_col='groups')

print(posthoc)

# -- d is number of reviews for the app
d = data['Reviews']

# -- R is average rating for the APP
R = data['Rating']

# -- m is min number of reviews to be listed in top 250
top250 = data.sort_values('Reviews',ascending=False)
top250 = top250[:250]
m = top250['Reviews'].iloc[249]

# -- C is the mean amount of reviews across whole
C = data['Reviews'].mean()

score = (d/(d+m))*R+(m/(d+m))
data['score'] = score
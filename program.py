import pandas as pd
import numpy as np
import datetime
import scipy
# import scikit_posthocs as sp
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
rank1 = category[category['Category'] == rankLst[0]]
rank2 = category[category['Category'] == rankLst[1]]
rank3 = category[category['Category'] == rankLst[2]]
rank4 = category[category['Category'] == rankLst[3]]
rank5 = category[category['Category'] == rankLst[4]]

stats.levene(rank1['Installs'],rank2['Installs']).pvalue

Anova = stats.kruskal(rank1['Installs'], rank2['Installs'], rank3['Installs'], rank4['Installs'], rank5['Installs'])
print(Anova)
print(Anova.pvalue)

# not done yet
# x_data = pd.DataFrame({'x1':rank1['Installs'], 'x2':rank2['Installs'], 'x3':rank3['Installs'], 'x4':rank4['Installs'], 'x5':rank5['Installs']})
# x_melt = pd.melt(x_data)
# posthoc = sp.posthoc_tamhane(
#     x_melt['value'], x_melt['variable'],
#     alpha=0.05)

# print(posthoc)

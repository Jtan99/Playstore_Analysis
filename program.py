import pandas as pd
import numpy as np
import datetime
import scipy
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from datetime import date
from scipy import stats

in_directory = 'googleplaystore.csv'

def to_timestamp(x):
    return x.timestamp()

def single_rating(rating):
    if rating<=1.7:
        return 'bad'
    elif rating<=3.3:
        return 'average'
    else:
        return 'excellent'

group_rating = np.vectorize(single_rating,
        otypes=[np.string_])

data = pd.read_csv(in_directory)
data = data.dropna()

data['Price'] = data['Price'].str.extract(r'(\d+.\d+)')[0].astype('float32')
data['Price'] = data['Price'].fillna(0)
data['Installs'] = data['Installs'].str.replace(r'[\+,]','').astype('int64')
data = data[data['Installs']>1000]

#parse the 'Last Updated' column to datetime type
data['Last Updated'] = pd.to_datetime(data['Last Updated'], format='%B %d, %Y')
data['Reviews'] = data['Reviews'].astype(int)

# --calculating a rank score --
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
# --rank score end --

paid = data[data['Type'] == "Paid"].copy()		# table of paid apps only
paid = paid[paid['Price']<100]
free = data[data['Type'] == "Free"].copy()

free['r_group'] = group_rating(free['Rating'])
paid['r_group'] = group_rating(paid['Rating'])

contingency = [[len(free[free['r_group'] == b'bad']),len(free[free['r_group'] == b'average']),len(free[free['r_group'] == b'excellent'])],
               [len(paid[paid['r_group'] == b'bad']),len(paid[paid['r_group'] == b'average']),len(paid[paid['r_group'] == b'excellent'])]]

chi2, p, dof, expected = stats.chi2_contingency(contingency)
print('Chi square p-value:',p)
# convert datetime column to int timestamp so we can plot it
# timestamp = data['Last Updated'].apply(to_timestamp)
paid_timestamp = paid['Last Updated'].apply(to_timestamp)

#LINEAR REGRESSION on price of apps compared to the last update time
#Do more recent/ recently updated apps cost more?
time_price_fit = stats.linregress(paid_timestamp,paid['Price'])
price_predict = paid_timestamp*time_price_fit.slope + time_price_fit.intercept

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)

plt.plot(paid['Last Updated'], paid['Price'], 'b.', alpha=0.5)
plt.plot(paid['Last Updated'], price_predict, 'r-', linewidth=3)
plt.xlabel('Last update')
plt.ylabel('Price')
plt.title('Change in Price over time')
# plt.show()
plt.savefig('Time vs Price')


time_rate_fit = stats.linregress(timestamp,data['Rating'])
rate_predict = timestamp*time_rate_fit.slope + time_rate_fit.intercept

plt.subplot(1,2,2)
plt.plot(data['Last Updated'], data['Rating'], 'b.', alpha=0.5)
plt.plot(data['Last Updated'], rate_predict, 'r-', linewidth=3)
plt.xlabel('Last update')
plt.ylabel('Rating')
plt.title('Change in Ratings over time')
# plt.show()
plt.savefig('Time vs Rating')

# null hyp = slope is zero
# Ha = slope is nonzero
# check if normal for OLS pvalue
rate_residuals = data['Rating'] - (time_rate_fit.slope*timestamp + time_rate_fit.intercept)
plt.hist(rate_residuals,25)

# print("pvalue for Price OLS test:", time_price_fit.pvalue)
# print("failed to reject null hypothesis\n")
print("pvalue for Rating OLS test:", time_rate_fit.pvalue)
print("strong evidence to reject null hypothesis")




category = data[['Category','App','score']].copy()
rank = category.groupby(['Category']).agg('count').sort_values('App', ascending = False)

# -- creating a bar graph for the market share --
bargraph = pd.DataFrame()
bargraph['Count'] = rank['App']
bargraph = bargraph.reset_index()
print(bargraph)

plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title('Market share bar graph')
plt.xlabel('category')
plt.ylabel('count')
plt.xticks(range(len(bargraph['Category'])), bargraph['Category'], rotation=90)
plt.tight_layout()
plt.bar(bargraph['Category'],bargraph['Count'])
# plt.show()
plt.savefig('marketShare')

# --getting the top 5 --
rankLst = rank.index.values.tolist()
rank1 = category[category['Category'] == rankLst[0]].reset_index()
rank2 = category[category['Category'] == rankLst[1]].reset_index()
rank3 = category[category['Category'] == rankLst[2]].reset_index()
rank4 = category[category['Category'] == rankLst[3]].reset_index()
rank5 = category[category['Category'] == rankLst[4]].reset_index()

# -- checking is the variance equal for the groups --
var = stats.levene(rank1['score'],rank2['score']).pvalue
print('Levene test for equal variance p-value:', var)

# -- doing an kruskal test, similar to anova but for unequal var and sample size
Kruskal = stats.kruskal(rank1['score'], rank2['score'], rank3['score'], rank4['score'], rank5['score'])
print(Kruskal)
# print(Kruskal.pvalue)

# -- Doing a posthoc tamhane test, similar to tukey but for unequal var and sample size --
x = pd.concat([rank1['score'],rank2['score'],rank3['score'],rank4['score'],rank5['score']], ignore_index=True, axis=1)
x = x.melt(var_name='groups', value_name='values')
posthoc = sp.posthoc_tamhane(x, val_col='values', group_col='groups')

print(posthoc)
X = sp.sign_array(posthoc).astype('int32')
print(X)	# Numpy array where 0 is False (not significant), 1 is True (significant)
plt.figure()
sp.sign_plot(X, flat = True)
plt.savefig('posthoc')

# -- Graph the visualization of the mean of each group and their confidence intervals
mean = x.groupby('groups')['values'].mean()
std = x.groupby('groups')['values'].std()

plt.figure()
plt.errorbar(mean.index, mean, xerr=0.5, yerr=2*std,fmt='o')
plt.title('Multiple Comparisons between All Pairs')
plt.ylabel('score')
plt.xlabel('Rank')
plt.savefig('Errorbar comparison')
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
print('\n\nChi square p-value:',p)
# convert datetime column to int timestamp so we can plot it
timestamp = data['Last Updated'].apply(to_timestamp)
paid_timestamp = paid['Last Updated'].apply(to_timestamp)

#LINEAR REGRESSION on price of apps compared to the last update time
#Do more recent/ recently updated apps cost more?
time_price_fit = stats.linregress(paid_timestamp,paid['Price'])
price_predict = paid_timestamp*time_price_fit.slope + time_price_fit.intercept

plt.figure(figsize=(15,5))
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


# -- can we use OLS test to check if slope is nonzero? need to check normality of residuals
rate_residuals = data['Rating'] - (time_rate_fit.slope*timestamp + time_rate_fit.intercept)
price_residuals = paid['Price'] - (time_price_fit.slope*paid_timestamp + time_price_fit.intercept)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(price_residuals)
plt.title('Price Residuals histogram')

plt.subplot(1,2,2)
plt.hist(rate_residuals,25)
plt.title('Ratings Residuals histogram')
plt.savefig('Residual normality check')

print("\n\npvalue for Rating OLS test:", time_rate_fit.pvalue)
print("strong evidence to reject null hypothesis: Ratings do change over time")


category = data[['Category','App','score']].copy()
group = category.groupby(['Category']).agg('count').sort_values('App', ascending = False)

# -- creating a bar graph for the market share --
bargraph = pd.DataFrame()
bargraph['Count'] = group['App']
bargraph = bargraph.reset_index()
print('\nMarket share table\n',bargraph,'\n')


plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title('Market share bar graph')
plt.xlabel('category')
plt.ylabel('count')
plt.xticks(range(len(bargraph['Category'])), bargraph['Category'], rotation=90)
plt.tight_layout()
plt.bar(bargraph['Category'],bargraph['Count'])
# plt.show()
plt.savefig('marketShare')

# --getting the largest 5 --
groupLst = group.index.values.tolist()
group1 = category[category['Category'] == groupLst[0]].reset_index()
group2 = category[category['Category'] == groupLst[1]].reset_index()
group3 = category[category['Category'] == groupLst[2]].reset_index()
group4 = category[category['Category'] == groupLst[3]].reset_index()
group5 = category[category['Category'] == groupLst[4]].reset_index()
group6 = category[category['Category'] == groupLst[5]].reset_index()
group7 = category[category['Category'] == groupLst[6]].reset_index()
group8 = category[category['Category'] == groupLst[7]].reset_index()

# -- checking is the variance equal for the groups --
var = stats.levene(group1['score'],group2['score']).pvalue
print('Levene test for equal variance p-value:', var)

# -- doing an kruskal test, similar to anova but non-parametric and unequal sample size
Kruskal = stats.kruskal(group1['score'], group2['score'], group3['score'], group4['score'],
                        group5['score'],group6['score'],group7['score'],group8['score'])
print('\n',Kruskal)
# print(Kruskal.pvalue)

# -- Doing a posthoc dunn test, after kruskal, works for non-parametric and unequal sample size --
x = [group1['score'].to_numpy(),group2['score'].to_numpy(),group3['score'].to_numpy(),group4['score'].to_numpy(),
     group5['score'].to_numpy(),group6['score'].to_numpy(),group7['score'].to_numpy(),group8['score'].to_numpy()]
posthoc = sp.posthoc_dunn(x, p_adjust = 'holm')

print('\nPosthoc p-value results\n',posthoc)
X = sp.sign_array(posthoc).astype('int32')
print('\nInterpretation of posthoc\nwhere 0 is False (not significant), 1 is True (significant)\n',X)
plt.figure()
sp.sign_plot(X, flat = True)    # Red is not significant diff, Green is significant diff
plt.savefig('posthoc')

# -- Graph the visualization of the mean of each group and their confidence intervals
x = pd.concat([group1['score'],group2['score'],group3['score'],group4['score'],
               group5['score'],group6['score'],group7['score'],group8['score']], ignore_index=True, axis=1)
x = x.melt(var_name='groups', value_name='values')
mean = x.groupby('groups')['values'].mean()
std = x.groupby('groups')['values'].std()

plt.figure()
plt.errorbar(mean.index, mean, xerr=0.5, yerr=2*std,fmt='o')
plt.title('Multiple Comparisons between All Pairs')
plt.ylabel('score')
plt.xlabel('Rank')
plt.savefig('Errorbar comparison')

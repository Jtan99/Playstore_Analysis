import pandas as pd
import numpy as np
import datetime
import scipy
import scikit_posthocs as sp
from datetime import date
from scipy import stats

in_directory = 'googleplaystore.csv'

data = pd.read_csv(in_directory)
data = data.dropna()

data['Price'] = data['Price'].str.extract(r'(\d+.\d+)')[0].astype('float32')
data['Price'] = data['Price'].fillna(0)

data['Last Updated'] = pd.to_datetime(data['Last Updated'], format='%B %d, %Y')

data['Installs'] = data['Installs'].str.replace(r'[\+,]','').astype('int64')
data = data[data['Installs']>1000]

paid = data[data['Type'] == "Paid"]

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
import pandas as pd
import numpy as np
import datetime
from datetime import date

in_directory = 'googleplaystore.csv'

def to_timestamp(x):
    return x.timestamp()

data = pd.read_csv(in_directory)
data = data.dropna()

data['Price'] = data['Price'].str.extract(r'(\d+.\d+)')[0].astype('float32')
data['Price'] = data['Price'].fillna(0)

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




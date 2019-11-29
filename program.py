import pandas as pd
import numpy as np
import datetime
from datetime import date

in_directory = 'googleplaystore.csv'

data = pd.read_csv(in_directory)
data = data.dropna()

data['Price'] = data['Price'].str.extract(r'(\d+.\d+)')[0].astype('float32')
data['Price'] = data['Price'].fillna(0)

data['Last Updated'] = pd.to_datetime(data['Last Updated'], format='%B %d, %Y')

paid = data[data['Type'] == "Paid"]
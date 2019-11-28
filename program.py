import pandas as pd
import numpy as np
import datetime
from datetime import date

in_directory = 'googleplaystore.csv'

data = pd.read_csv(in_directory,parse_dates=['Last Updated'])

data['Price'] = data['Price'].str.extract(r'(\d+.\d+)')[0].astype('float32')
data['Price'] = data['Price'].fillna(0)

paid = data[data['Type'] == "Paid"]
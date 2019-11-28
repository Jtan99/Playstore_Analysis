import pandas as pd
import numpy as np

def main(in_directory):
	data = pd.read_csv(in_directory)

 if __name__=='__main__':
	in_directory = sys.argv[1]
	# out_directory = sys.argv[2]
	main(in_directory)
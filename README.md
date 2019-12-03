# CMPT-353-Project
For this project, our topic of research will involve a dataset of information about mobile apps installed in the Google Play Store. The objective of our research is to find any general trends that can potentially be used to help make business decisions. There is a variety of possible questions to explore, but for this project, our main focus will be on the following questions:
 
1. Are mobile apps getting more popular/well-received over the years? Are apps getting more expensive?
2. Is there any correlation between the price of an app and the average rating? 
3. What are the ranks of each app and is there a particular category that dominates the top app rankings?


Program requirements:

python 3.x

If you do not have the following libraries installed, Please enter the corresponding commands

scikit-posthocs 0.6.1

	pip install scikit-posthocs

scipy 1.3.3

	pip install scipy

Start up:
To run the program enter the following command

	python program.py input

If the program compiles correctly, there should be 4 saved figures:  


	Regression Graphs.png
	Two graphs showing the regression graph of Price vs Time and Rating vs Time  
	
	Residual normality check.png
	Two histograms showing the distribution of the residuals, to check OLS assumption  
	
	marketShare.png
	A histogram of the amount of apps on the dataset based on their categories  
	
	posthoc and Errorbar comparison.png
	The results of our Dunn test: a significance plot, and a errorbar graph  
	

import pandas
import seaborn
import numpy
import matplotlib.pyplot as pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DataFrame = pandas.read_csv('ecommerce.csv')


print DataFrame.columns
print DataFrame.head()
print DataFrame.describe()

seaborn.jointplot(DataFrame['Time on Website'],DataFrame['Yearly Amount Spent'])
seaborn.jointplot(DataFrame['Time on App'],DataFrame['Yearly Amount Spent'])
seaborn.pairplot(DataFrame)

pyplot.show()


LinearReg = LinearRegression()

X = DataFrame[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = DataFrame['Yearly Amount Spent']

XTrainingSet, XTestingSet, YTrainingSet, YTestingSet = train_test_split( X, y, test_size = 0.35, random_state = 90)

LinearReg.fit(XTrainingSet, YTrainingSet)

LinearRegPredict = LinearReg.predict(XTestingSet)




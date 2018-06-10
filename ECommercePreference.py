import pandas
import seaborn
import numpy
import matplotlib.pyplot as pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

DataFrame = pandas.read_csv('ecommerce.csv')


print DataFrame.columns
print DataFrame.head()
print DataFrame.describe()



seaborn.jointplot(DataFrame['Time on Website'],DataFrame['Yearly Amount Spent'])
seaborn.jointplot(DataFrame['Time on App'],DataFrame['Yearly Amount Spent'])
seaborn.pairplot(DataFrame)




LinearReg = LinearRegression()

X = DataFrame[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = DataFrame['Yearly Amount Spent']

XTrainingSet, XTestingSet, YTrainingSet, YTestingSet = train_test_split( X, y, test_size = 0.35, random_state = 90)

LinearReg.fit(XTrainingSet, YTrainingSet)

LinearRegPredict = LinearReg.predict(XTestingSet)

Coefficients = LinearReg.coef_
### Intercept = LinearReg.intercept_

pyplot.scatter( YTestingSet, LinearRegPredict)


print metrics.mean_absolute_error(YTestingSet, LinearRegPredict)
print metrics.mean_squared_error(YTestingSet, LinearRegPredict)
print numpy.sqrt(metrics.mean_squared_error(YTestingSet, LinearRegPredict))

CoefficientsDF = pandas.DataFrame(Coefficients, X.columns, columns = ['Coefficients'])

seaborn.distplot(LinearRegPredict)

print CoefficientsDF

pyplot.show()
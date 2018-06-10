import pandas
import seaborn
import numpy
import matplotlib.pyplot as pyplot


DataFrame = pandas.read_csv('ecommerce.csv')

print DataFrame.columns
print DataFrame.head()
print DataFrame.describe()

seaborn.jointplot(DataFrame['Time on Website'],DataFrame['Yearly Amount Spent'])
seaborn.jointplot(DataFrame['Time on App'],DataFrame['Yearly Amount Spent'])
seaborn.pairplot(DataFrame)

pyplot.show()
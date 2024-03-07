import pandas as pd
insurance = pd.read_csv('insurance.csv')

#----------q1a: check for multicollinearity
print(insurance.corr())
#no high collinearity found

#----------q1b: recode the variable
import numpy as np
insurance['east_west'] = np.where((insurance['region']=='southwest') | (insurance['region']=='northwest'), 1, 0)
#recode the categorical variables while we're at it
dummies = pd.get_dummies(insurance[['sex','smoker']])
insurance = pd.concat([insurance,dummies],axis=1)
insurance.head()

#----------q1c
import statsmodels.formula.api as sm
model1a = sm.ols('charges~east_west', data = insurance).fit()
print(model1a.summary())

#----------q1d: 
model1b = sm.ols('charges~east_west+age',data=insurance).fit()
print(model1b.summary()) #east_west stays significant
model1c = sm.ols('charges~east_west+sex_female',data=insurance).fit()
print(model1c.summary()) #east_west stays significant
model1d = sm.ols('charges~east_west+bmi',data=insurance).fit()
print(model1d.summary()) #east_west is no longer significant
model1e = sm.ols('charges~east_west+children',data=insurance).fit()
print(model1e.summary()) #east_west stays significant
model1f = sm.ols('charges~east_west+smoker_yes',data=insurance).fit()
print(model1f.summary()) #east_west is no longer significant

#table showing averages split up by east_west show that the bmi in the west is lower
#and that fewer people smoke in the west
print(insurance.groupby('east_west').mean())


#----------q2
#first import the file
cpu = pd.read_csv('cpu.csv')
cpu.head()

#----------q2a
model2a = sm.ols('CpuTime ~ CardsIn + LinesOut + Steps + MountedDevices', data=cpu).fit()
print(model2a.summary())
#the R-squared is very high (>0.9) so we can use it for predictions

#----------q2b: CardsIn is not significant so we will take it out
model2b = sm.ols('CpuTime ~ LinesOut + Steps + MountedDevices', data=cpu).fit()
print(model2b.summary())

#----------q2c: predict the first row
cpupred = model2b.predict() #create the predictions
print(cpupred) #print all the predictions
print(cpupred[0]) #print just the first prediction

#compare the prediction with the real value:
print("the prediction is off by", cpu.CpuTime[0] - cpupred[0])
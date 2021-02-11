ReadMe
#Variable names to be changed:
flowrate = pd.read_csv('FRO_HC1_.csv', usecols=[2, 3])

flowrate_threshold = 2

X_test = X.loc['2013-01-01':'2013-12-31'].values
X = X.loc['1992-01-01':'2013-01-01'].values#Changed

classifier = load('DecisionTreeForLSTM_FRO_HC1.joblib')
dump(classifier, 'DecisionTreeForLSTM_FRO_HC1.joblib')


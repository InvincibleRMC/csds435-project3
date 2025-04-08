from surprise import Dataset
from surprise.accuracy import rmse

from surprise.model_selection import KFold
from surprise.prediction_algorithms import SVD

from accuracy import accuracy

data = Dataset.load_builtin("ml-100k")

param_grid = {'n_factors': 50, 'n_epochs': 50, 'lr_all': 0.005, 'reg_all': 0.4}
gs = SVD(**param_grid)

for i, (train_set, test_set) in enumerate(KFold(5).split(data)):
    gs.fit(train_set)

    predicted = gs.test(test_set)
    rmse(predicted)
    accuracy(predicted)


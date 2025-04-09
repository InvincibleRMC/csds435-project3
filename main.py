from surprise import Dataset
from surprise.accuracy import rmse

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithZScore


from accuracy import accuracy

data = Dataset.load_builtin("ml-100k")

param_grid = {
    "k": 40,
    "min_k": 1,
    "sim_options": {"name": "msd", "min_support": 1, "user_based": False},
    "verbose": False,
}

gs = KNNWithZScore(**param_grid)

for i, (train_set, test_set) in enumerate(KFold(5).split(data)):
    gs.fit(train_set)

    predicted = gs.test(test_set)
    rmse(predicted)
    accuracy(predicted)

from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise import Dataset
from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin("ml-100k")

# Training
param_grid = {
    "k": [5, 10, 20, 40],
    "min_k": [1, 3, 5, 7],
    "sim_options": {
        "name": ["msd", "cosine"],
        "min_support": [1, 5, 7],
        "user_based": [False, True],
    },
    "verbose": [False],
}

gs = GridSearchCV(KNNWithZScore, param_grid, measures=["rmse", "mae"], cv=5)

gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])

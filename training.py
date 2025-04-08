from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin("ml-100k")

# Training
param_grid = {
    "n_factors": [50, 100, 200],
    "n_epochs": [5, 10, 15, 25, 50],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.4, 0.6, 1.0],
}
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5)

gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])

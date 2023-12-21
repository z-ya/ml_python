from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

digits_data = datasets.load_digits()

images_features = digits_data.images.reshape((len(digits_data.images), -1))
images_target = digits_data.target

random_forest_clf = RandomForestClassifier(n_jobs=-1, max_features='sqrt')
features_train, features_test, target_train, target_test = train_test_split(images_features, images_target, test_size=0.3)

param_grid = {
    'n-estimators': [10, 100, 500, 1000],
    'max_depth':[1, 5, 10, 15],
    'min_samples_leaf':[1, 2, 4, 10, 15, 30, 50]
}

grid_search = GridSearchCV(random_forest_clf, param_grid, cv=10)
grid_search.fit(features_train, target_train)
print(grid_search.best_params_)

optimal_estimators = grid_search.best_params_.get('n_estimators')
optimal_depth = grid_search.best_params_.get('max_depth')
optimal_leaf = grid_search.best_params_.get('min_samples_leaf')





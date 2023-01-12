import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

df = pd.read_csv('FODS-A2.csv')
# print(df.columns)

target = 'Appliances'
X = df.drop(target,axis=1)
y = df[target]

data_train, data_test = train_test_split(df, test_size = 0.1, random_state = 0)

def f(x):
    pca = PCA(n_components=x)
    model = pca.fit(X)
    X_pc = model.transform(X)
    n_pcs= model.components_.shape[0]
    imp = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
    features = X.columns
    imp_features = [features[imp[i]] for i in range(n_pcs)]
    dic = {'PC{}'.format(i): imp_features[i] for i in range(n_pcs)}
    df = pd.DataFrame(dic.items())
    if x>25:
        print(df)
        print("\n")
    return round(sum(list(pca.explained_variance_ratio_))*100, 2)
# print(f(4))

rx = np.array(range(1,27))
ry = [f(x) for x in rx]

plt.plot(rx, ry, '-o')
for index, value in enumerate(ry):
    plt.text(index+0.5, value+0.5, str(value), fontsize = 8)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Run standardization on X variables
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# Define cross-validation folds
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Linear Regression
lin_reg = LinearRegression().fit(X_train_scaled, y_train)
lr_score_train = -1 * cross_val_score(lin_reg, X_train_scaled, y_train, cv=cv, scoring='neg_root_mean_squared_error').mean()
lr_score_test = mean_squared_error(y_test, lin_reg.predict(X_test_scaled), squared=False)

# Generate all the principal components
pca = PCA() # Default n_components = min(n_samples, n_features)
X_train_pc = pca.fit_transform(X_train_scaled)

print(pca.explained_variance_ratio_)

# Initialize linear regression instance
lin_reg = LinearRegression()

# Create empty list to store RMSE for each iteration
rmse_list = []

# Loop through different count of principal components for linear regression
for i in range(1, X_train_pc.shape[1]+1):
    rmse_score = -1 * cross_val_score(lin_reg, 
                                      X_train_pc[:,:i], # Use first k principal components
                                      y_train, 
                                      cv=cv, 
                                      scoring='neg_root_mean_squared_error').mean()
    rmse_list.append(rmse_score)

# rmse_list = [x/1000 for x in rmse_list]
# lr_score_train = [x/1000 for x in lr_score_train]

# Visual analysis - plot RMSE vs count of principal components used
plt.plot(rmse_list, '-o')
plt.xlabel('Number of principal components in regression')
plt.ylabel('RMSE')
plt.title('Quality')
plt.xlim(xmin=-1);
plt.xticks(np.arange(X_train_pc.shape[1]), np.arange(1, X_train_pc.shape[1]+1))
plt.axhline(y=lr_score_train, color='g', linestyle='-');
plt.show()

from sklearn.model_selection import train_test_split
import pandas as pd
import torch

X = pd.DataFrame([[2, 0, 4], [10, 4, 7], [5, 3, 2], [12, 13, 14]])
y = pd.DataFrame([5, 12, 3, 15])
print(X)
print('')
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
X_tr2, X_tes2, y_tr2, y_tes2 = train_test_split(X_test, y_test, test_size=0.5)
print(X_train)
print(' ')
print(X_tes2)

ix = torch.randint(100 - 5, (5,))
print(ix)

# need to randomly get/remove val and test data
# need to replace those values in train data with <Reserved>
# if beginning with <Reserved> shift right until no longer
# if ending with <Reserved> shift left until no longer



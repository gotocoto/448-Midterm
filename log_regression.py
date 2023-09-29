from sklearn.linear_model import LogisticRegression

# Assume X, y are your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train, y_train)

# Making predictions
y_pred_lr = lr_classifier.predict(X_test)

# Evaluating the model
print(classification_report(y_test, y_pred_lr))

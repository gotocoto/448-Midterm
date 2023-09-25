from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume X, y are your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Making predictions
y_pred_nb = nb_classifier.predict(X_test)

# Evaluating the model
print(classification_report(y_test, y_pred_nb))

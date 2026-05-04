from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    precision = precision_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')

    report = classification_report(y_test, preds)

    return acc, f1, precision, recall, report, preds
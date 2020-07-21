import numpy as np
import pandas as pd
import pickle
with open('ensemble_nn.pkl', 'rb') as handle:
    new_model = pickle.load(handle)
with open('test_features.pkl', 'rb') as handle:
    test_features = pickle.load(handle)
with open('dict.pkl', 'rb') as handle:
    di = pickle.load(handle)
predictions = new_model.predict(test_features)
pred_labels = np.argmax(predictions, axis = 1)
print(pred_labels)
print(di)
# print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels))) 


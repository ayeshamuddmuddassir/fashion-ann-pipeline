import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

model = keras.models.load_model("models/model.h5")
te = np.load("data/processed/test.npz")

loss, acc = model.evaluate(te["x"], te["y"], verbose=0)
pred = np.argmax(model.predict(te["x"], verbose=0), axis=1)

os.makedirs("plots", exist_ok=True)
cm = confusion_matrix(te["y"], pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.savefig("plots/confusion_matrix.png")

json.dump({"test_loss": float(loss), "test_accuracy": float(acc)},
          open("metrics.json", "w"), indent=2)
print("Test accuracy:", acc)

import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]
d = np.load("data/raw/fashion_raw.npz")

x_train = d["x_train"].astype("float32") / 255.0
x_test = d["x_test"].astype("float32") / 255.0

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, d["y_train"],
    test_size=params["test_size"], random_state=params["seed"],
    stratify=d["y_train"])

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed("data/processed/train.npz", x=x_tr, y=y_tr)
np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
np.savez_compressed("data/processed/test.npz", x=x_test, y=d["y_test"])
print("Processed:", x_tr.shape, x_val.shape, x_test.shape)
# WIP edit

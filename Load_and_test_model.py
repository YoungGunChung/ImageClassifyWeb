import os
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("mpg_model.h5")
x = np.zeros((1, 7))

x[0, 0] = 8  # 'cylinders',
x[0, 1] = 400  # 'displacement',
x[0, 2] = 80  # 'horsepower',
x[0, 3] = 2000  # 'weight',
x[0, 4] = 19  # 'acceleration',
x[0, 5] = 72  # 'year',
x[0, 6] = 1  # 'origin'


pred = model.predict(x)
print(float(pred[0]))

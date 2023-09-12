import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label = "sin")
plt.plot(x, y2, linestyle = "--", label = "cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('/content/drive/MyDrive/Colab Notebooks/밑바닥딥러닝/PhotoRoom_20221117_031206.png')
plt.imshow(img)
plt.show()
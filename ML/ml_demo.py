"""
This is a quick demo of deep speckle correlation project.

Paper link: https://arxiv.org/abs/1806.04139

Author: Yunzhe Li, Yujia Xue, Lei Tian

Computational Imaging System Lab, @ ECE, Boston University

Date: 2018.08.21
"""
import matplotlib.pyplot as plt
import numpy as np

from model2 import get_model_deep_speckle


# model is defined in model.py
model = get_model_deep_speckle()
# pretrained_weights.hdf5 can be downloaded from the link on our GitHub project page
model.load_weights('.binarystar.50.hdf5')

# test speckle patterns. Four types of objects (E,S,8,9),
# Each object has five speckle patterns through 5 different test diffusers
#speckle_E = np.load('test_data/letter_E.npy')
#speckle_S = np.load('test_data/letter_S.npy')
#speckle_8 = np.load('test_data/number_8.npy')
#speckle_9 = np.load('test_data/test.npy')
for i in range(0,1):
    f1 = f"test/x_{i}.npy"
    f2 = f"test/c_{i}.npy"


    test = np.array([np.load(f1)])
    # prediction
    pred = model.predict(test, batch_size=2)
    print(pred)
    result = np.load(f2)
    print(result)
#pred_speckle_S = model.predict(speckle_S, batch_size=2)
#pred_speckle_8 = model.predict(speckle_8, batch_size=2)
#pred_speckle_9 = model.predict(speckle_9, batch_size=2)
#print(pred_speckle_8.shape)
# plot results
"""
plt.figure()
plt.imshow(pred_speckle_E[0], cmap='hot')
plt.show()"""
"""
plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(speckle_E[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_speckle_E[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(speckle_S[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_speckle_S[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(speckle_8[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_speckle_8[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.figure()
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(speckle_9[i, :].squeeze(), cmap='hot')
    plt.axis('off')
    plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(pred_speckle_9[i, :, :, 0].squeeze(), cmap='gray')
    plt.axis('off')

plt.show()
"""
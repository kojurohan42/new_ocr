import os
import cv2
import copy
import numpy as np

from scipy import stats
from keras.models import load_model

Models = np.array([load_model(os.path.join(Path, 'best_val_loss.hdf5')) for Path in ['Model_1', 'Model_2', 'Model_3']])

def predit_core(image):
    if image.shape[1] < image.shape[0] - 10:
        pad_width = (image.shape[0] - 10 - image.shape[1]) // 2
        padded_image = np.pad(image, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
    else:
        padded_image = image

    # plt.imshow(padded_image,)
    # plt.show()
    print("------------------------")
    thresh = cv2.resize(padded_image, (32, 32), interpolation = cv2.INTER_AREA)
    # plt.imshow(thresh)
    # plt.show()
    x = np.array([thresh]).reshape(-1, 32, 32, 1) / 255.0
    y = np.array([np.argmax(Model.predict(x)) for Model in Models])
    print("core Modifiers",y)
    return y
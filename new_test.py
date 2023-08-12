import numpy as np
import cv2
# import tensorflow as tf
import os

import Split_Words
import Split_Characters
import Predict_Characters

img = cv2.imread("new.jpg")

Words = Split_Words.Split(img)
Split_Characters.Split(Words)


# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()


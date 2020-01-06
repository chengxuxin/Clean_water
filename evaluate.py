import numpy as np
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import random

model = load_model('model1.h5')

test_x_orig = np.load('test_input.npy')
test_y = np.load('test_label.npy')
test_x = test_x_orig.reshape((-1, 30, 40, 1))
test_x = test_x.astype('float32')
test_x /= 255
# test_y = to_categorical(test_y, 11)

for i in range(20):
    index = random.randint(0, 330)
    pred_one_hot = model.predict(np.reshape(test_x[index, :, :, :], (1, 30, 40, 1)))
    pred = np.argmax(pred_one_hot)
    # print(pred, '\n', test_y[i])
    plt.figure(figsize=(3, 2.25))
    img = test_x_orig[index, :, :].astype(int)
    plt.imshow(img, cmap='gray')
    plt.title('true:%s predicted:%s' % (int(test_y[index]), pred))
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

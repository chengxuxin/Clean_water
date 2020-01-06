import random
import numpy as np
import cv2 as cv

train_index = np.zeros(770)
test_index = np.zeros(330)
for i in range(11):
    random_num_train = random.sample(range(0, 100), 70)
    random_num_test = tuple(set(range(0, 100)) - set(random_num_train))
    train_index[i * 70:(i + 1) * 70] = np.asarray(random_num_train) + i * 100
    test_index[i * 30:(i + 1) * 30] = np.asarray(random_num_test) + i * 100

train_index = train_index.astype(int)
test_index = test_index.astype(int)

train_input = np.empty((770, 30, 40))
train_label = np.empty(770)
count = 0
for i in train_index:
    img = cv.imread('/Users/cxx/Desktop/Clean_water/CleanWaterDataset/' +
                    'clean' + str(i//100*10) + '/' + str(i).zfill(3) + '.png')
    resized = cv.resize(img, (40, 30))
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    train_input[count, :, :] = gray
    train_label[count] = 10 - i//100
    count += 1
    print(count)

np.save('train_input', train_input)
np.save('train_label', train_label)

test_input = np.empty((330, 30, 40))
test_label = np.empty(330)
count = 0
for i in test_index:
    img = cv.imread('/Users/cxx/Desktop/Clean_water/CleanWaterDataset/' +
                    'clean' + str(i//100*10) + '/' + str(i).zfill(3) + '.png')
    resized = cv.resize(img, (40, 30))
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    test_input[count, :, :] = gray
    test_label[count] = 10 - i//100
    count += 1
    print(count)

np.save('test_input', test_input)
np.save('test_label', test_label)
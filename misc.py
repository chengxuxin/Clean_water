import cv2 as cv


#
img = cv.imread('/Users/cxx/Desktop/000.png')
print(img.shape)
print(img.size)
print(img.dtype)
resized = cv.resize(img, (40, 30))
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
print(img.shape)
print(img.size)
print(img.dtype)

# cv.namedWindow("testtt")
cv.imshow('test', gray)
cv.waitKey(0)

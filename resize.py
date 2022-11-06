import cv2
img_path = './images/warrior4.jpg'
fx_fy = 0.65

img = cv2.imread(img_path)
print('Image Width is', img.shape[1])
print('Image Height is', img.shape[0])
img_resize = cv2.resize(img, None, fx=fx_fy, fy=fx_fy)
print('Image Width is', img_resize .shape[1])
print('Image Height is', img_resize .shape[0])

cv2.imwrite(img_path, img_resize)

cv2.imshow('CLASSIFICATION OF YOGA POSE', img_resize)
# cv2.imshow('CLASSIFICATION OF YOGA POSE', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
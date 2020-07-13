import cv2
cam = cv2.VideoCapture(0)
print("----- current ----")
print(cam.get(cv2.CAP_PROP_FPS))
print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("----- setting ----")
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret_val, image = cam.read()
print('cam image=%dx%d' % (image.shape[1], image.shape[0]))

while True:
  ret_val, image = cam.read()
  cv2.imshow('test', image)
  if cv2.waitKey(1) == 27:
    break
cv2.destroyAllWindows()


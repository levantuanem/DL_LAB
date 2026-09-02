import cv2
image = cv2.imread(r'D:\AI_Projects\DL_LAB\data\raw\images\effen.jpg') #BGR
#Lấy ra từng kênh màu 1 
image[:,:,1] = 0 # blue
image[:,:,2] = 0
print(image.shape) # chiều dọc-> ngang -> sâu
cv2.imshow('image', image)
cv2.waitKey(0)
# cv2.destroyAllWindows()
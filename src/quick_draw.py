import numpy as np
import matplotlib.pyplot as plt
# Bước 1: Chọn file 
file_path = r"D:\AI_Projects\DL_LAB\data\raw\Quickdraw\full_numpy_bitmap_bicycle.npy"
images = np.load(file_path).astype(np.float32)
# # print(images.shape)
train_images = images[:-10] # Lấy tất cả ảnh ngoại trừ 10 ảnh cuối ra làm bộ training
test_images = images[-10:] # Giữ 10 bức ảnh cuối làm bộ test

# # Bước 2: Tính ra bức ảnh trung bình của bộ training, kích thước 28x28 pixel
# avg_image = np.mean(train_images, axis = 0)
# # avg_image = np.reshape(avg_image, (28, 28))
# # print(avg_image.shape)

# # Bước 3: Visualize bức ảnh trung bình => Quan sát xem có nhận ra được category không
# # plt.imshow(avg_image)
# # plt.show()

# # Bước 4: Chọn 1 index từ 0 - 9
# # Sau đó tính tích vô hướng (dot product) của bức ảnh trung bình và ảnh test
# index = 4
# test_images = test_images[index]
# # print(test_images.shape)
# score1 = np.dot(train_images, avg_image) # cách 1 dùng dot numpy
# score2 = (test_images @ avg_image) # cách 2 dùng dấu @
# score3 = np.matmul(test_images, avg_image) # cách 3 dùng matmul của numpy(chỉ đúng khi nó là mảng 1 chiều)
# # print(score1)
# # print(score2)
# # print(score3)

# ===========================================================
# ===========================================================
# ===========================================================
# ===========================================================

# LÀM TƯƠNG TỰ VỚI CÁC CATEGORY CÒN LẠI. Sau đó tính tích vô hướng của từng ảnh
# trung bình của ảnh test chọn ở bước 4 với từng bức ảnh trung bình này
categories = ["bicycle", 
              "binoculars", 
              "bird", 
              "birthday cake", 
              "blackberry", 
              "blueberry",
              "book",
              "boomerang", 
              "bottlecap",
              "bowtie"]
avg_images = []
scores = []
for c in categories:
    file_path = r"D:\AI_Projects\DL_LAB\data\raw\Quickdraw\full_numpy_bitmap_{}.npy".format(c)
    images = np.load(file_path).astype(np.float32)
    avg_image = np.mean(images, axis = 0)
    avg_images.append(avg_image.reshape((28,28)))
    dot_product = (test_images @ avg_image)
    scores.append(dot_product)
for score in scores:
    print(score)
# Cuối cùng các bạn xem là liệu trong 10 score này, score tương ứng với tích vô hướng
# của ảnh test này với ảnh trung bình của category của chính nó có phải là score 
# lớn nhất. Bức ảnh trung bình tính ra có thể xem như là weight cho từng category

# Bước 6: Visualize 10 weight (avg_image) này trong cùng 1 ảnh kích thước 2x5 hoặc 5x2
# để so sánh xem  weight của các cotegories nào dễ nhìn và weight nào không 
plt.figure(figsize = (10, 4))
for i in range(len(categories)):
    plt.subplot(2, 5, i+1)
    plt.imshow(avg_images[i])
    plt.title(categories[i])
    plt.tight_layout()
plt.show()
## 1. BẢN CHẤT CỦA DEEP LEARNING TRONG XỬ LÝ ẢNH
*   **Sự khác biệt cốt lõi:** Thay vì phải tự trích xuất đặc trưng (Feature Extraction) thủ công như Machine Learning truyền thống, Deep Learning gộp chung bước trích xuất đặc trưng và phân loại (Classification) vào cùng một mạng Nơ-ron để máy tự học.
*   **Cách máy tính nhìn nhận ảnh:** Ảnh màu được hiểu là một ma trận 3 chiều (Width x Height x 3 kênh màu RGB). Ảnh xám là ma trận 2 chiều.
*   **3 Bài toán chính:**
    *   **Classification:** Bức ảnh này chứa cái gì? (Phân loại toàn bộ ảnh).
    *   **Object Detection:** Các đối tượng nằm ở đâu (Bounding Box) và là gì?
    *   **Segmentation:** Khoanh vùng chính xác từng pixel của đối tượng (Phân đoạn).

## 2. KIẾN TRÚC MẠNG NƠ-RON (NEURAL NETWORKS)
*   **Linear Classifier (Phân loại tuyến tính):** Dựa trên hàm $f(x, W, b) = Wx + b$. Điểm hạn chế là chỉ giải quyết được các bài toán tuyến tính.
*   **Nơ-ron nhân tạo:** Lấy cảm hứng từ não bộ. Input đi qua các trọng số (Weights) -> Tính tổng $z = \sum w_i x_i + b$ -> Đi qua **Hàm kích hoạt (Activation Function)** để tạo ra tính phi tuyến, giúp mạng học được các đặc trưng phức tạp.
*   **Các Hàm Kích Hoạt Quan Trọng:**
    *   **Sigmoid:** Giới hạn đầu ra từ [0, 1]. Dễ bị Vanishing Gradient.
    *   **Tanh:** Giới hạn đầu ra từ [-1, 1]. Tốt hơn Sigmoid nhưng vẫn bị Vanishing.
    *   **ReLU:** $g(z) = \max(0, z)$. Phổ biến nhất, tính toán cực nhanh, giảm thiểu Vanishing Gradient nhưng có thể gây hiện tượng "Dead ReLU" (nơ-ron không bao giờ kích hoạt nếu đầu vào âm).
    *   **Leaky ReLU:** Trị số âm vẫn có độ dốc nhỏ để tránh Dead ReLU.

## 3. CƠ CHẾ HỌC CỦA MÔ HÌNH (TRAINING PIPELINE)
Quá trình học là việc điều chỉnh các trọng số (Weights) sao cho sai số (Loss) là nhỏ nhất.
*   **Hàm Mất mát (Loss Function):**
    *   *Hồi quy (Regression):* MAE, MSE, Huber Loss.
    *   *Phân loại (Classification):* Cross-Entropy Loss (thường kết hợp với Softmax ở output layer để chuyển logit thành xác suất tổng bằng 1).
*   **Tối ưu hóa (Optimization - Gradient Descent):**
    *   Mục tiêu là đi tìm đáy của thung lũng (Global minimum). Cập nhật trọng số theo hướng ngược lại của đạo hàm (Gradient).
    *   **Thách thức:** Dễ kẹt ở Local minima (đáy giả) hoặc Saddle points (điểm yên ngựa). Nếu Learning rate quá nhỏ thì học chậm, quá lớn thì phân kỳ không hội tụ được.
    *   **Giải pháp:** 
        *   Thêm **Momentum** (quán tính) để vượt qua các vũng lõm nhỏ.
        *   Các bộ tối ưu thích ứng (Adaptive) như **Adagrad, RMSprop** tự động điều chỉnh learning rate.
*   **Backpropagation (Lan truyền ngược):** Kỹ thuật dùng quy tắc chuỗi (Chain Rule) để tính đạo hàm ngược từ đầu ra về đầu vào, qua đó biết cần cập nhật trọng số mỗi lớp bao nhiêu.

## 4. MẠNG TÍCH CHẬP (CNN) - TRÁI TIM CỦA COMPUTER VISION
Tại sao không dùng mạng thẳng (Fully Connected - FC) cho ảnh? Vì quá nhiều tham số và làm mất thông tin không gian (không tính đến sự liên kết giữa các pixel lân cận). CNN giải quyết bằng **Kết nối cục bộ (Local Connectivity)** và **Chia sẻ tham số (Parameter Sharing)**.
*   **Các khối cấu trúc của CNN:**
    *   **Convolution Layer (Tích chập):** Dùng các bộ lọc (Kernels/Filters) trượt qua ảnh để trích xuất đặc trưng (cạnh, góc, texture). Các tham số cần nhớ: Khung trượt (Stride - $s$), Đệm viền (Padding - $p$).
    *   **Pooling Layer (Lớp gộp):** Thường dùng Max Pooling. Chức năng chính là giảm kích thước (Downsampling), giảm số lượng tham số, và giúp mô hình có khả năng bất biến với sự dịch chuyển nhẹ (Translation Invariance).
    *   **1x1 Convolution:** Dùng để giảm số lượng kênh (channels), tiết kiệm chi phí tính toán.
*   **Receptive Field:** Vùng trên ảnh gốc ảnh hưởng đến 1 nơ-ron ở lớp hiện tại. Để mạng "nhìn" được bối cảnh rộng hơn, cần tăng Receptive Field bằng cách: Thêm Conv layer, dùng Pooling, hoặc dùng Dilated Convolution.

## 5. CÁC KỸ THUẬT NÂNG CAO TRONG HUẤN LUYỆN
*   **Vấn đề Vanishing / Exploding Gradients:** Trọng số bị triệt tiêu về 0 hoặc bùng nổ lên vô cực ở các mạng sâu.
    *   *Cách khắc phục:* Khởi tạo trọng số chuẩn (Xavier cho Sigmoid/Tanh, He initialization cho ReLU). Cắt xén gradient (Gradient Clipping).
*   **Batch Normalization:** Chuẩn hóa dữ liệu ở mỗi mini-batch ngay bên trong mạng (trừ đi mean, chia cho variance). Giúp mô hình hội tụ nhanh hơn hẳn, cho phép dùng Learning rate lớn hơn.
*   **Dropout:** Tắt ngẫu nhiên một tỷ lệ nơ-ron trong lúc train. Đây là kỹ thuật Regularization chống Overfitting cực kỳ hiệu quả.
*   **Data Augmentation:** Xoay, lật, cắt, đổi màu ảnh gốc để sinh ra dữ liệu mới, giúp mô hình tổng quát hóa tốt hơn (không học vẹt).
*   **Transfer Learning:** Lấy một mô hình đã train sẵn trên tập dữ liệu khổng lồ (như ImageNet), sau đó dùng làm bộ trích xuất đặc trưng hoặc "fine-tune" lại một chút để áp dụng cho bài toán cụ thể của mình. Cực kỳ tiết kiệm thời gian và tài nguyên.

## 6. OBJECT DETECTION (ĐỊNH VỊ ĐỐI TƯỢNG)
*   **Hệ sinh thái R-CNN (Region-based CNN):**
    *   **R-CNN:** Chạy thuật toán Selective Search tìm ra khoảng 2000 vùng nghi ngờ -> Đưa từng vùng qua mạng CNN -> Chạy SVM phân loại. (Rất chậm và nặng).
    *   **Fast R-CNN:** Đưa toàn bộ ảnh qua mạng CNN 1 lần duy nhất để lấy Feature Map -> Chiếu các vùng nghi ngờ (từ Selective Search) lên Feature Map -> Dùng RoI Pooling cắt đặc trưng -> Phân loại. (Nhanh hơn đáng kể).
    *   **Faster R-CNN:** Bỏ hẳn thuật toán Selective Search chậm chạp bên ngoài. Đưa hẳn một mạng phụ gọi là **RPN (Region Proposal Network)** vào bên trong mạng chính để máy tự học cách đề xuất vùng (thông qua các **Anchors**). Toàn bộ mô hình chạy End-to-end cực kỳ tối ưu.
*   **Các khái niệm đánh giá cốt lõi:**
    *   **IoU (Intersection over Union):** Tỷ lệ diện tích giao nhau chia cho diện tích hợp nhau giữa Hộp dự đoán và Hộp thực tế. IoU > 0.5 (hoặc 0.7) thường được coi là dự đoán trúng.
    *   **NMS (Non-maximum Suppression):** Một đối tượng có thể bị dự đoán nhiều hộp chồng lên nhau. NMS sẽ giữ lại hộp có độ tự tin cao nhất và xóa các hộp xung quanh bị trùng lặp (có IoU cao với hộp giữ lại).
    *   **mAP (Mean Average Precision):** Thang đo chuẩn mực nhất để đánh giá một mô hình Object Detection mạnh hay yếu.
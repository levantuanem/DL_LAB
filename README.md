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

# 2.8. Backpropagation

## 2.8.1. Backpropagation là gì?

**Backpropagation (Lan truyền ngược)** là thuật toán được sử dụng để tính toán **gradient của Loss Function đối với các parameters (Weight và Bias)** trong Neural Network.

Backpropagation giúp Neural Network biết được:

> Mỗi Weight và Bias đã ảnh hưởng đến Loss như thế nào?

Quá trình training Neural Network có thể mô tả:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Function
  ↓
Backpropagation
  ↓
Gradient
  ↓
Optimizer
  ↓
Update Parameters
```

Backpropagation **không trực tiếp cập nhật Weight và Bias**.

Nó chỉ tính toán gradient:

```text
∂L/∂W
∂L/∂b
```

Sau đó Optimizer như Gradient Descent, Momentum, RMSProp hoặc Adam sử dụng các gradient này để cập nhật parameters.

---

## 2.8.2. Tại sao cần Backpropagation?

Một Neural Network có thể chứa hàng nghìn, hàng triệu hoặc thậm chí hàng tỷ parameters.

Ví dụ:

```text
Input Layer
     ↓
Hidden Layer 1
     ↓
Hidden Layer 2
     ↓
Hidden Layer 3
     ↓
Output Layer
```

Mỗi layer chứa nhiều Weight và Bias.

Trong quá trình training, chúng ta cần biết Loss thay đổi như thế nào khi thay đổi từng parameter:

```text
∂L/∂W1
∂L/∂W2
∂L/∂W3
...
∂L/∂Wn
```

Trong đó:

```text
L = Loss
W = Weight
```

Gradient cho biết:

> Nếu thay đổi một parameter một lượng rất nhỏ thì Loss sẽ thay đổi theo hướng và mức độ như thế nào.

Backpropagation giúp tính toán tất cả các gradient này một cách hiệu quả.

---

# 2.8.3. Ý tưởng cốt lõi của Backpropagation

Backpropagation dựa trên **Chain Rule (Quy tắc chuỗi)** trong Calculus.

Nếu:

```text
y = f(u)
u = g(x)
```

thì:

```text
dy/dx = dy/du × du/dx
```

Trong Neural Network, Loss phụ thuộc vào Output, Output phụ thuộc vào Hidden Layer, Hidden Layer lại phụ thuộc vào Weight và Bias.

Ví dụ:

```text
Weight
   ↓
z
   ↓
Activation
   ↓
a
   ↓
Output
   ↓
Loss
```

Do đó để tính:

```text
∂L/∂W
```

ta sử dụng Chain Rule để nhân các đạo hàm trên toàn bộ chuỗi.

Ví dụ:

```text
W → z → a → ŷ → L
```

thì:

```text
∂L/∂W
=
∂L/∂ŷ
×
∂ŷ/∂a
×
∂a/∂z
×
∂z/∂W
```

Đây chính là cơ chế toán học cốt lõi của Backpropagation.

---

# 2.8.4. Forward Propagation

Trước khi thực hiện Backpropagation, Neural Network phải thực hiện Forward Propagation.

Với một layer:

```text
z = Wa + b
```

Trong đó:

```text
W = Weight
a = Activation từ layer trước
b = Bias
z = Pre-activation
```

Sau đó áp dụng Activation Function:

```text
a = f(z)
```

Ví dụ với ReLU:

```text
ReLU(z) = max(0, z)
```

Quá trình tiếp tục qua các layer cho đến Output:

```text
Input
  ↓
z1 → a1
  ↓
z2 → a2
  ↓
z3 → a3
  ↓
ŷ
  ↓
Loss
```

---

# 2.8.5. Backward Propagation

Sau khi tính được Loss, quá trình Backpropagation bắt đầu.

Thay vì đi từ Input đến Output, gradient được truyền ngược từ Output về Input.

```text
Forward:

Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Output
  ↓
Loss
```

Backpropagation:

```text
Loss
  ↓
Output
  ↓
Layer 3
  ↓
Layer 2
  ↓
Layer 1
```

Tại mỗi layer, Backpropagation tính gradient của Loss đối với:

```text
Weight
Bias
Activation
```

Sau đó gradient được truyền tiếp về layer trước.

---

# 2.8.6. Ví dụ Backpropagation đơn giản

Xét Neural Network:

```text
x
 ↓
w1
 ↓
z1
 ↓
ReLU
 ↓
a1
 ↓
w2
 ↓
ŷ
 ↓
Loss
```

Các công thức:

```text
z1 = w1x + b1
```

```text
a1 = ReLU(z1)
```

```text
ŷ = w2a1 + b2
```

Giả sử sử dụng MSE dạng:

```text
L = 1/2(ŷ - y)²
```

Ta muốn tính:

```text
∂L/∂w1
```

Loss không phụ thuộc trực tiếp vào `w1`.

Nó phụ thuộc theo chuỗi:

```text
w1
 ↓
z1
 ↓
a1
 ↓
ŷ
 ↓
L
```

Do đó sử dụng Chain Rule:

```text
∂L/∂w1
=
∂L/∂ŷ
×
∂ŷ/∂a1
×
∂a1/∂z1
×
∂z1/∂w1
```

---

# 2.8.7. Gradient của Output Layer

Giả sử:

```text
L = 1/2(ŷ - y)²
```

Đạo hàm Loss theo Prediction:

```text
∂L/∂ŷ = ŷ - y
```

Với:

```text
ŷ = w2a1 + b2
```

ta có:

```text
∂ŷ/∂w2 = a1
```

Do đó:

```text
∂L/∂w2
=
∂L/∂ŷ × ∂ŷ/∂w2
```

Suy ra:

```text
∂L/∂w2
=
(ŷ - y)a1
```

Gradient của Bias:

```text
∂ŷ/∂b2 = 1
```

nên:

```text
∂L/∂b2
=
∂L/∂ŷ
```

---

# 2.8.8. Gradient của Hidden Layer

Để tính gradient cho `w1`:

```text
w1
 ↓
z1
 ↓
a1
 ↓
ŷ
 ↓
L
```

Áp dụng Chain Rule:

```text
∂L/∂w1
=
∂L/∂ŷ
×
∂ŷ/∂a1
×
∂a1/∂z1
×
∂z1/∂w1
```

Ta có:

```text
∂ŷ/∂a1 = w2
```

Với:

```text
a1 = ReLU(z1)
```

thì:

```text
∂a1/∂z1 = ReLU'(z1)
```

Và:

```text
z1 = w1x + b1
```

nên:

```text
∂z1/∂w1 = x
```

Do đó:

```text
∂L/∂w1
=
∂L/∂ŷ
×
w2
×
ReLU'(z1)
×
x
```

---

# 2.8.9. Gradient của Bias

Tương tự:

```text
∂L/∂b1
=
∂L/∂ŷ
×
∂ŷ/∂a1
×
∂a1/∂z1
×
∂z1/∂b1
```

Vì:

```text
∂z1/∂b1 = 1
```

nên:

```text
∂L/∂b1
=
∂L/∂ŷ
×
w2
×
ReLU'(z1)
```

---

# 2.8.10. Delta / Error Signal

Trong nhiều tài liệu về Neural Network, ta thường gặp ký hiệu:

```text
δ
```

được gọi là **Delta** hoặc **Error Signal**.

Delta thường được định nghĩa:

```text
δ = ∂L/∂z
```

Với:

```text
a = f(z)
```

ta có:

```text
δ
=
∂L/∂a × f'(z)
```

Sau khi tính được `δ`, gradient của Weight có thể viết:

```text
∂L/∂W = δaᵀ
```

Gradient của Bias:

```text
∂L/∂b = δ
```

Gradient truyền về layer trước:

```text
∂L/∂a_previous = Wᵀδ
```

Ba công thức này là những công thức quan trọng trong Backpropagation.

---

# 2.8.11. Backpropagation trong Neural Network nhiều Layer

Với mạng:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Output
  ↓
Loss
```

Forward Propagation:

```text
a0
 ↓
z1 → a1
 ↓
z2 → a2
 ↓
z3 → a3
 ↓
ŷ
 ↓
L
```

Backpropagation:

```text
L
 ↓
∂L/∂ŷ
 ↓
∂L/∂z3
 ↓
∂L/∂W3
 ↓
∂L/∂a2
 ↓
∂L/∂z2
 ↓
∂L/∂W2
 ↓
∂L/∂a1
 ↓
∂L/∂z1
 ↓
∂L/∂W1
```

Gradient được truyền từ layer cuối về các layer đầu.

---

# 2.8.12. Công thức tổng quát cho một Layer

Một layer có:

```text
z = Wa + b
```

và:

```text
a = f(z)
```

Giả sử đã biết:

```text
∂L/∂z
```

Ta có:

### Gradient của Weight

```text
∂L/∂W
=
∂L/∂z × aᵀ
```

### Gradient của Bias

```text
∂L/∂b
=
∂L/∂z
```

### Gradient truyền về layer trước

```text
∂L/∂a
=
Wᵀ × ∂L/∂z
```

Đây là cơ sở để thực hiện Backpropagation cho Neural Network nhiều layer.

---

# 2.8.13. Backpropagation và Activation Function

Activation Function đóng vai trò quan trọng vì Backpropagation phải tính đạo hàm của Activation Function.

Ví dụ ReLU:

```text
ReLU(z) = max(0,z)
```

Đạo hàm:

```text
ReLU'(z) =
    1  nếu z > 0
    0  nếu z < 0
```

Nếu:

```text
z > 0
```

gradient được truyền qua.

Nếu:

```text
z < 0
```

gradient trở thành:

```text
0
```

Các Activation Function khác cũng có đạo hàm tương ứng:

```text
Sigmoid
Tanh
ReLU
Leaky ReLU
Softmax
```

Backpropagation sử dụng các đạo hàm này để truyền gradient qua mạng.

---

# 2.8.14. Vanishing Gradient

Một vấn đề quan trọng của Backpropagation là **Vanishing Gradient**.

Gradient được tính bằng Chain Rule:

```text
∂L/∂W1
=
gradient_n
×
gradient_(n-1)
×
...
×
gradient_1
```

Nếu các gradient có giá trị nhỏ hơn 1, việc nhân nhiều gradient có thể khiến giá trị giảm rất nhanh:

```text
0.5 × 0.5 × 0.5 × 0.5 × ...
```

Kết quả:

```text
Gradient ≈ 0
```

Khi đó các layer gần Input nhận được gradient rất nhỏ.

Hậu quả:

```text
Weight update rất nhỏ
        ↓
Layer đầu học rất chậm
        ↓
Training khó khăn
```

Một số giải pháp:

```text
ReLU
Leaky ReLU
Proper Weight Initialization
Batch Normalization
Residual Connections
```

---

# 2.8.15. Exploding Gradient

Ngược lại, nếu gradient quá lớn:

```text
2 × 2 × 2 × 2 × ...
```

gradient có thể tăng rất nhanh.

Hiện tượng này gọi là:

**Exploding Gradient.**

Khi đó:

```text
Gradient → rất lớn
```

làm cho parameter update quá lớn và quá trình training trở nên không ổn định.

Một kỹ thuật thường dùng để xử lý là:

```text
Gradient Clipping
```

Ngoài ra có thể sử dụng:

```text
Proper Weight Initialization
Normalization
Appropriate Learning Rate
```

---

# 2.8.16. Computational Graph

Một cách trực quan để hiểu Backpropagation là sử dụng **Computational Graph**.

Ví dụ:

```text
x
│
▼
× w
│
▼
z
│
▼
ReLU
│
▼
a
│
▼
× w2
│
▼
ŷ
│
▼
Loss
```

Forward:

```text
x → z → a → ŷ → L
```

Backward:

```text
∂L/∂ŷ
   ↓
∂L/∂a
   ↓
∂L/∂z
   ↓
∂L/∂w
```

Mỗi phép toán trong Computational Graph đều có một local gradient.

Backpropagation kết hợp các local gradient bằng Chain Rule để tính gradient từ Loss về từng parameter.

---

# 2.8.17. Automatic Differentiation

Trong các framework Deep Learning hiện đại như:

```text
PyTorch
TensorFlow
JAX
```

chúng ta không cần tự tính từng đạo hàm bằng tay.

Framework có thể tự động tính gradient.

Ví dụ trong PyTorch:

```python
loss.backward()
```

Lệnh này thực hiện quá trình Backpropagation và tính gradient cho các parameters.

Sau đó:

```python
optimizer.step()
```

sử dụng gradient để cập nhật parameters.

Quy trình:

```text
loss.backward()
      ↓
Backpropagation
      ↓
Calculate Gradients
      ↓
optimizer.step()
      ↓
Update Parameters
```

---

# 2.8.18. Backpropagation ≠ Gradient Descent

Hai khái niệm này cần được phân biệt rõ.

| Backpropagation               | Gradient Descent        |
| ----------------------------- | ----------------------- |
| Tính Gradient                 | Cập nhật Parameters     |
| Sử dụng Chain Rule            | Sử dụng Gradient        |
| Lan truyền Gradient ngược     | Tối ưu Parameters       |
| Tính `∂L/∂W`, `∂L/∂b`         | Tạo `W_new`, `b_new`    |
| Không trực tiếp update Weight | Trực tiếp update Weight |

Quá trình đầy đủ:

```text
Forward Propagation
        ↓
Calculate Prediction
        ↓
Calculate Loss
        ↓
Backpropagation
        ↓
Calculate Gradient
        ↓
Optimizer
        ↓
Update Parameters
```

---

# 2.8.19. Backpropagation + Gradient Descent

Gradient Descent sử dụng gradient do Backpropagation tính:

```text
W_new = W_old - η × ∂L/∂W
```

```text
b_new = b_old - η × ∂L/∂b
```

Trong đó:

```text
η = Learning Rate
```

Ví dụ:

```text
W = 4
∂L/∂W = 63
η = 0.01
```

thì:

```text
W_new
=
4 - 0.01 × 63
=
3.37
```

Backpropagation trả lời:

> Gradient là bao nhiêu?

Gradient Descent trả lời:

> Dùng gradient đó để thay đổi parameter như thế nào?

---

# 2.8.20. Một vòng Training hoàn chỉnh

Một vòng training của Neural Network:

```text
1. Initialize Parameters
        ↓
2. Forward Propagation
        ↓
3. Calculate Prediction
        ↓
4. Calculate Loss
        ↓
5. Backpropagation
        ↓
6. Calculate Gradients
        ↓
7. Optimizer
        ↓
8. Update Parameters
        ↓
9. Repeat
```

Ví dụ:

```text
Epoch 1
Loss = 2.50
     ↓
Backpropagation
     ↓
Calculate Gradients
     ↓
Update Parameters

Epoch 2
Loss = 1.80
     ↓
Backpropagation
     ↓
Calculate Gradients
     ↓
Update Parameters

Epoch 3
Loss = 1.20
     ↓
...
```

Mục tiêu của quá trình training:

```text
Loss ↓
```

---

# 2.8.21. Tổng kết

Backpropagation là thuật toán giúp Neural Network **tính gradient của Loss đối với các parameters**.

Ý tưởng chính:

```text
Forward Propagation
Input → Prediction → Loss
```

Sau đó:

```text
Backpropagation
Loss → Gradient
```

Cuối cùng:

```text
Optimizer
Gradient → Updated Parameters
```

Backpropagation dựa trên **Chain Rule**:

```text
∂L/∂W
=
∂L/∂output
×
∂output/∂...
×
...
×
∂.../∂W
```

Đối với một layer:

```text
z = Wa + b
```

ta có:

```text
∂L/∂W = δaᵀ
```

```text
∂L/∂b = δ
```

```text
∂L/∂a = Wᵀδ
```

Trong đó:

```text
δ = ∂L/∂z
```

Có thể ghi nhớ toàn bộ quá trình bằng:

```text
                FORWARD
                   ↓
Input → Layers → Prediction → Loss
                              ↓
                         BACKPROPAGATION
                              ↓
                          Gradients
                              ↓
                           OPTIMIZER
                              ↓
                       Update Parameters
                              ↓
                           Repeat
```

**Backpropagation = tính gradient bằng cách lan truyền ngược Chain Rule qua Computational Graph.**

**Gradient Descent/Optimizer = sử dụng gradient đó để cập nhật parameters và giảm Loss.**

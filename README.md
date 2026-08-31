````markdown
# 📚 Deep Learning
## 1. Tổng quan về Deep Learning
- Deep Learning (Học sâu) là một nhánh của Machine Learning, sử dụng các mạng nơ-ron nhân tạo nhiều lớp (Deep Neural Networks - DNN) để máy tính tự động học các đặc trưng và mối quan hệ phức tạp từ dữ liệu.
- Deep Learning đặc biệt hiệu quả với các loại dữ liệu lớn và phức tạp như:
    + 🖼️ Hình ảnh
    + 🎵 Âm thanh
    + 📝 Văn bản
    + 📈 Time Series
    + 📹 Video
    + 📡 Sensor Data
---
### 1.1. Vị trí của Deep Learning trong AI
```text
Artificial Intelligence (AI)
│
└── Machine Learning (ML)
    │
    └── Deep Learning (DL)
        │
        ├── Neural Network
        ├── CNN
        ├── RNN / LSTM / GRU
        ├── Transformer
        └── Autoencoder
```
### 1.2. Deep Learning
- Deep Learning là một nhánh của Machine Learning sử dụng Neural Network nhiều tầng.
```text
AI
 ↓
Machine Learning
 ↓
Deep Learning
 ↓
Deep Neural Network
```
- Điểm mạnh của Deep Learning là khả năng tự học feature từ dữ liệu.
---
## 2. Neural Network là gì?
- Neural Network là mô hình tính toán lấy cảm hứng từ cách hoạt động của neuron trong hệ thần kinh.
- Một mạng Neural Network cơ bản gồm:
```text
Input Layer
     ↓
Hidden Layer
     ↓
Hidden Layer
     ↓
Output Layer
```
- Ví dụ:
```text
Input
 │
 ├── x₁
 ├── x₂
 ├── x₃
 │
 ↓
Hidden Layer
 │
 ├── Neuron
 ├── Neuron
 └── Neuron
 │
 ↓
Output
```
- Khi mạng có nhiều Hidden Layer, chúng ta gọi là:
> Deep Neural Network (DNN)
---
### 2.1. Cấu trúc của Neural Network
- Một Neural Network thường gồm 3 loại layer chính:
```text
Input Layer
     ↓
Hidden Layers
     ↓
Output Layer
```
### 2.2. Input Layer
- Nhận dữ liệu đầu vào.
- Ví dụ bài toán dự đoán điểm:
```text
reading_score
writing_score
gender
lunch
test_preparation
```
- Hoặc bài toán Computer Vision:
```text
Image
 ↓
Pixels
 ↓
Input Layer
```
---
### 2.3. Hidden Layer
- Hidden Layer thực hiện các phép tính và học feature từ dữ liệu.
- Ví dụ:
```text
Input
 ↓
Hidden Layer 1
 ↓
Hidden Layer 2
 ↓
Hidden Layer 3
 ↓
Output
```
- Các layer càng sâu có thể học các feature càng phức tạp.
---
### 2.4. Output Layer
- Output Layer tạo kết quả cuối cùng.
- Ví dụ Classification:
```text
Input Image
     ↓
Neural Network
     ↓
Output
 ┌─────────────┐
 │ Cat   0.10  │
 │ Dog   0.85  │
 │ Car   0.05  │
 └─────────────┘
```
- Model dự đoán:
```text
Dog
```
---
## 3. Neuron
- Neuron là đơn vị tính toán cơ bản của Neural Network.
- Một neuron nhận nhiều input:
```text
x₁ ──┐
     │
x₂ ──┼──> Neuron ──> Output
     │
x₃ ──┘
```
- Mỗi input có một **Weight** tương ứng.
```text
x₁ ── w₁ ──┐
x₂ ── w₂ ──┼──> Neuron
x₃ ── w₃ ──┘
```
- Neuron tính:
$$
z = \sum_{i=1}^{n} w_i x_i + b
$$
- Sau đó đưa qua Activation Function:
$$
y = f(z)
$$
- Trong đó:
    + $x_i$: Input
    + $w_i$: Weight
    + $b$: Bias
    + $z$: Tổng có trọng số
    + $f$: Activation Function
    + $y$: Output
---
## 4. Weight

**Weight** thể hiện mức độ ảnh hưởng của một input đến neuron.

Ví dụ:

```text
Input 1 ── Weight 1 ──┐
                      │
Input 2 ── Weight 2 ──┼──> Neuron
                      │
Input 3 ── Weight 3 ──┘
```

Trong quá trình training, model liên tục điều chỉnh các weight để giảm lỗi.

---

## 5. Bias

Bias giúp điều chỉnh đầu ra của neuron.

Công thức:

$$
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$

Trong đó:

```text
w = Weight
x = Input
b = Bias
```

Weight và Bias là những **trainable parameters** của Neural Network.

---

## 6. Activation Function

Activation Function quyết định output của neuron.

Nếu không có activation function, nhiều layer tuyến tính kết hợp lại vẫn chỉ tạo thành một hàm tuyến tính.

Activation Function giúp Neural Network học được các quan hệ **phi tuyến** phức tạp.

---

### 6.1. ReLU

ReLU là activation function rất phổ biến trong Deep Learning.

$$
ReLU(x) = max(0,x)
$$

Ví dụ:

```text
x = -5 → 0
x = -1 → 0
x =  0 → 0
x =  2 → 2
x =  5 → 5
```

Ưu điểm:

- Đơn giản
- Tính toán nhanh
- Giảm vấn đề Vanishing Gradient so với Sigmoid/Tanh trong nhiều mạng sâu

---

### 6.2. Sigmoid

Công thức:

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

Output nằm trong khoảng:

```text
0 → 1
```

Thường được sử dụng trong:

- Binary Classification
- Output probability

---

### 6.3. Tanh

Output nằm trong khoảng:

```text
-1 → 1
```

Công thức:

$$
tanh(x)
$$

---

### 6.4. Softmax

Softmax thường được sử dụng ở Output Layer của bài toán **Multi-class Classification**.

Ví dụ:

```text
Cat       → 0.10
Dog       → 0.75
Car       → 0.15
```

Tổng xác suất:

```text
0.10 + 0.75 + 0.15 = 1.0
```
---

# 9. Forward Propagation

**Forward Propagation** là quá trình dữ liệu đi từ Input → Hidden Layers → Output.

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
```

Ví dụ:

```text
Input
x₁, x₂, x₃
   ↓
Weighted Sum
   ↓
Activation
   ↓
Hidden Layer
   ↓
Weighted Sum
   ↓
Activation
   ↓
Output
```

Mục tiêu của Forward Propagation là tạo ra:

> **Prediction**

---

# 10. Loss Function

Sau khi model dự đoán, cần xác định model dự đoán tốt hay xấu.

Loss Function đo độ sai giữa:

```text
Actual Value
     vs
Predicted Value
```

Ví dụ:

```text
Actual = 100
Prediction = 90
```

Model có error.

Loss Function biến error thành một giá trị số.

```text
Loss nhỏ  → Prediction tốt
Loss lớn  → Prediction kém
```

---

# 11. Một số Loss Function

## Regression

### Mean Squared Error - MSE

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2
$$

Trong đó:

- $y_i$: Giá trị thực
- $\hat{y_i}$: Giá trị dự đoán

---

## Classification

Một loss function phổ biến là:

**Cross Entropy Loss**

Binary Classification thường sử dụng:

```text
Binary Cross Entropy
```

Multi-class Classification thường sử dụng:

```text
Categorical Cross Entropy
```

---

# 12. Backpropagation

**Backpropagation** là quá trình tính gradient của Loss đối với các parameter trong mạng.

```text
Forward
   ↓
Prediction
   ↓
Loss
   ↓
Backpropagation
   ↓
Gradient
```

Gradient cho biết:

> Weight cần thay đổi theo hướng nào để Loss giảm.

---

# 13. Gradient Descent

Gradient Descent là một phương pháp tối ưu để tìm các parameter giúp giảm Loss.

Công thức đơn giản:

$$
w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}
$$

Trong đó:

- $w$: Weight
- $L$: Loss
- $\eta$: Learning Rate
- $\frac{\partial L}{\partial w}$: Gradient

Quá trình:

```text
Weight
  ↓
Calculate Loss
  ↓
Calculate Gradient
  ↓
Update Weight
  ↓
Loss giảm
  ↓
Lặp lại
```

---

# 14. Optimizer

Optimizer chịu trách nhiệm cập nhật các parameter của model dựa trên gradient.

Một số Optimizer phổ biến:

- SGD
- Momentum
- RMSprop
- Adam
- AdamW

Trong thực tế, **Adam/AdamW** thường được sử dụng rộng rãi vì khả năng hội tụ tốt trên nhiều bài toán.

---

# 15. Learning Rate

Learning Rate quyết định mức độ thay đổi của Weight trong mỗi lần update.

```text
Learning Rate quá lớn
        ↓
Training không ổn định
        ↓
Loss có thể dao động

Learning Rate quá nhỏ
        ↓
Training rất chậm
```

Có thể hình dung:

```text
Learning Rate
      ↓
Weight Update
      ↓
Training Speed
```

---

# 16. Batch và Epoch

## Batch

Batch là một nhóm dữ liệu được đưa vào model trong một lần training.

Ví dụ:

```text
Dataset = 10,000 samples
Batch Size = 100
```

Một epoch sẽ có:

```text
10,000 / 100 = 100 batches
```

---

## Epoch

Một **Epoch** là một lần model đi qua toàn bộ Training Dataset.

Ví dụ:

```text
Dataset
 ↓
Batch 1
 ↓
Batch 2
 ↓
Batch 3
 ↓
...
 ↓
Batch 100
 ↓
1 Epoch
```

Nếu training:

```text
Epoch = 50
```

thì model sẽ đi qua toàn bộ dataset 50 lần.

---

# 17. Quá trình Training tổng thể

Toàn bộ quá trình Deep Learning Training:

```text
Dataset
   ↓
Preprocessing
   ↓
Train / Validation / Test
   ↓
Batch
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
   ↓
Next Batch
   ↓
Next Epoch
   ↓
Evaluation
```

---

# 18. Train / Validation / Test

Dataset thường được chia thành:

```text
Dataset
│
├── Training Set
│
├── Validation Set
│
└── Test Set
```

### Training Set

Dùng để train model.

### Validation Set

Dùng để:

- Điều chỉnh hyperparameter
- Theo dõi training
- Chọn model
- Early stopping

### Test Set

Dùng để đánh giá model cuối cùng trên dữ liệu chưa từng được sử dụng trong quá trình training.

---

# 19. Overfitting và Underfitting

## Overfitting

Model học quá tốt Training Dataset nhưng hoạt động kém trên dữ liệu mới.

```text
Training Performance
       ↓
      Rất tốt

Validation Performance
       ↓
      Kém
```

Nguyên nhân:

- Model quá phức tạp
- Dataset nhỏ
- Training quá lâu
- Noise trong dữ liệu

Giải pháp:

- Dropout
- Data Augmentation
- Regularization
- Early Stopping
- Thu thập thêm dữ liệu
- Giảm độ phức tạp model

---

## Underfitting

Model quá đơn giản và chưa học được pattern trong dữ liệu.

```text
Training Performance
       ↓
      Kém

Validation Performance
       ↓
      Kém
```

Giải pháp:

- Tăng model capacity
- Train lâu hơn
- Feature tốt hơn
- Điều chỉnh Learning Rate

---

# 20. CNN - Convolutional Neural Network

CNN là kiến trúc Deep Learning được sử dụng rất nhiều trong **Computer Vision**.

Kiến trúc cơ bản:

```text
Image
 ↓
Convolution
 ↓
Activation
 ↓
Pooling
 ↓
Convolution
 ↓
Activation
 ↓
Pooling
 ↓
Flatten
 ↓
Fully Connected
 ↓
Output
```

---

# 21. Convolution

Convolution sử dụng một **Kernel/Filter** để quét qua hình ảnh.

```text
Image
  ↓
Filter / Kernel
  ↓
Convolution
  ↓
Feature Map
```

CNN có thể học các feature như:

```text
Layer nông
 ↓
Edge
 ↓
Texture
 ↓
Shape
 ↓
Object Parts
 ↓
Object
```

---

# 22. Pooling

Pooling giúp giảm kích thước Feature Map.

Hai loại phổ biến:

- Max Pooling
- Average Pooling

Ví dụ:

```text
Feature Map
     ↓
Max Pooling
     ↓
Smaller Feature Map
```

Lợi ích:

- Giảm số lượng computation
- Giảm kích thước feature
- Giúp model có tính bất biến nhất định với vị trí

---

# 23. CNN Applications

CNN được sử dụng trong:

- Image Classification
- Object Detection
- Semantic Segmentation
- Face Recognition
- OCR
- Medical Imaging
- Autonomous Driving
- ADAS
- Industrial Vision

Một số model nổi tiếng:

- LeNet
- AlexNet
- VGG
- ResNet
- MobileNet
- EfficientNet
- YOLO

---

# 24. RNN

**RNN - Recurrent Neural Network** được thiết kế cho dữ liệu tuần tự.

Ví dụ:

```text
x₁ → RNN → h₁
          ↓
x₂ → RNN → h₂
          ↓
x₃ → RNN → h₃
          ↓
x₄ → RNN → h₄
```

RNN có khả năng sử dụng thông tin từ các bước trước đó.

Ứng dụng:

- Time Series
- Speech
- Text
- Sensor Data

---

# 25. LSTM

**LSTM - Long Short-Term Memory** là một biến thể của RNN.

LSTM được thiết kế để xử lý vấn đề phụ thuộc dài hạn tốt hơn RNN truyền thống.

Các thành phần chính:

```text
Forget Gate
Input Gate
Output Gate
Cell State
```

Ứng dụng:

- Time Series Forecasting
- Speech
- NLP
- Sensor Forecasting

---

# 26. GRU

**GRU - Gated Recurrent Unit** cũng là một biến thể của RNN.

GRU có cấu trúc đơn giản hơn LSTM nhưng vẫn có khả năng ghi nhớ thông tin dài hạn.

So sánh:

| RNN | LSTM | GRU |
|---|---|---|
| Đơn giản | Phức tạp hơn | Đơn giản hơn LSTM |
| Dễ gặp Vanishing Gradient | Giải quyết tốt hơn | Giải quyết tốt hơn |
| Ít parameter | Nhiều parameter | Ít hơn LSTM |
| Sequence | Sequence | Sequence |

---

# 27. Transformer

Transformer là một trong những kiến trúc quan trọng nhất của Deep Learning hiện đại.

Kiến trúc tổng quát:

```text
Input
 ↓
Embedding
 ↓
Positional Encoding
 ↓
Self-Attention
 ↓
Feed Forward
 ↓
Transformer Block
 ↓
Output
```

Thành phần quan trọng:

- Embedding
- Self-Attention
- Multi-Head Attention
- Feed Forward Network
- Positional Encoding

---

# 28. Self-Attention

Self-Attention giúp model xác định mức độ liên quan giữa các phần khác nhau của input.

Ví dụ câu:

```text
The car stopped because it was damaged.
```

Model cần hiểu:

```text
"it"
 ↓
"car"
```

Attention giúp model học mối quan hệ giữa các token.

---

# 29. Ứng dụng Transformer

Transformer được sử dụng trong:

- NLP
- Machine Translation
- Text Generation
- LLM
- Computer Vision
- Multimodal AI
- Speech

Các hệ thống AI hiện đại dựa nhiều vào kiến trúc Transformer.

---

# 30. Machine Learning vs Deep Learning

| Machine Learning | Deep Learning |
|---|---|
| Thường cần Feature Engineering | Có khả năng tự học Feature |
| Phù hợp dữ liệu vừa/nhỏ | Mạnh với dữ liệu lớn |
| Model thường đơn giản hơn | Model có nhiều layer |
| Training thường nhanh hơn | Training thường nặng hơn |
| Ít tài nguyên hơn | Có thể cần GPU/TPU |
| Random Forest | CNN |
| SVM | RNN |
| XGBoost | Transformer |

---

# 31. Ví dụ Feature Engineering

## Machine Learning truyền thống

```text
Raw Image
    ↓
Feature Engineering
    ↓
HOG
Edge Detection
Texture
    ↓
SVM
    ↓
Prediction
```

Con người phải quyết định feature nào cần sử dụng.

---

## Deep Learning

```text
Raw Image
    ↓
CNN
    ↓
Feature Learning
    ↓
Classification
```

CNN tự học feature thông qua quá trình training.

---

# 32. Deep Learning cho Time Series

Deep Learning có thể được sử dụng cho dữ liệu Time Series.

Ví dụ dự báo CO₂:

```text
CO₂(t-10)
CO₂(t-9)
CO₂(t-8)
...
CO₂(t-1)
CO₂(t)
      ↓
Deep Learning Model
      ↓
CO₂(t+1)
```

Có thể sử dụng:

- RNN
- LSTM
- GRU
- 1D CNN
- Transformer

---

# 33. Deep Learning cho Sensor Data

Trong Embedded và IoT, sensor tạo ra dữ liệu liên tục:

```text
Temperature
Humidity
Pressure
CO₂
Accelerometer
Gyroscope
GPS
```

Có thể xây dựng:

```text
Sensor
 ↓
Data Collection
 ↓
Preprocessing
 ↓
Deep Learning
 ↓
Prediction
 ↓
Decision
```

Ví dụ:

```text
Accelerometer
      ↓
LSTM / CNN
      ↓
Detect Machine Fault
```

---

# 34. Deep Learning trong Edge AI

**Edge AI** là việc thực hiện AI inference trực tiếp tại thiết bị Edge thay vì gửi toàn bộ dữ liệu lên Cloud.

Kiến trúc:

```text
Sensor / Camera
       ↓
Edge Device
       ↓
Preprocessing
       ↓
AI Model
       ↓
Inference
       ↓
Decision
       ↓
Actuator
```

Ví dụ:

```text
Camera
   ↓
YOLO
   ↓
Vehicle Detection
   ↓
Embedded System
   ↓
ADAS Decision
```

---

# 35. Cloud AI vs Edge AI

| Cloud AI | Edge AI |
|---|---|
| Xử lý trên Cloud | Xử lý tại thiết bị |
| Cần Internet trong nhiều trường hợp | Có thể hoạt động Offline |
| Có tài nguyên tính toán lớn | Tài nguyên hạn chế |
| Latency phụ thuộc mạng | Latency thấp |
| Dễ triển khai model lớn | Cần tối ưu model |
| Privacy có thể phức tạp hơn | Dữ liệu có thể xử lý local |

---

# 36. TinyML

**TinyML** là lĩnh vực triển khai Machine Learning/Deep Learning trên các thiết bị có tài nguyên rất hạn chế.

Ví dụ:

```text
ESP32
Arduino
STM32
Microcontroller
```

Kiến trúc:

```text
Sensor
 ↓
Microcontroller
 ↓
TinyML Model
 ↓
Inference
 ↓
Action
```

Ví dụ:

```text
Accelerometer
 ↓
TinyML
 ↓
Gesture Recognition
 ↓
Control Device
```

---

# 37. Tại sao Deep Learning cần tối ưu khi chạy trên Edge?

Deep Learning Model có thể có:

- Hàng triệu parameters
- FLOPs lớn
- RAM lớn
- Flash lớn
- Latency cao
- Power consumption cao

Trong khi Microcontroller thường có:

```text
RAM hạn chế
Flash hạn chế
CPU yếu hơn GPU
Power hạn chế
```

Do đó cần tối ưu model.

---

# 38. Quantization

Quantization chuyển parameter từ dạng precision cao sang precision thấp.

Ví dụ:

```text
FP32
 ↓
INT8
```

So sánh:

```text
FP32 → 32-bit
INT8 → 8-bit
```

Lợi ích:

- Model nhỏ hơn
- Inference nhanh hơn
- Giảm Memory
- Giảm Power Consumption

INT8 Quantization đặc biệt quan trọng trong Edge AI.

---

# 39. Pruning

Pruning loại bỏ những parameter ít quan trọng.

```text
Original Model
      ↓
Remove unnecessary weights
      ↓
Smaller Model
```

Mục tiêu:

- Giảm model size
- Giảm computation
- Tăng tốc inference

---

# 40. Knowledge Distillation

Knowledge Distillation sử dụng một model lớn:

```text
Teacher Model
      ↓
Knowledge
      ↓
Student Model
```

Teacher có thể rất lớn và chính xác.

Student nhỏ hơn nhưng cố gắng học kiến thức từ Teacher.

```text
Large Model
    ↓
Knowledge Distillation
    ↓
Small Model
    ↓
Edge Device
```

---

# 41. ONNX

**ONNX - Open Neural Network Exchange** là format giúp trao đổi model giữa các framework.

Ví dụ:

```text
PyTorch
   ↓
ONNX
   ↓
Inference Runtime
```

ONNX thường được sử dụng trong workflow deployment.

---

# 42. TensorRT

**TensorRT** là SDK của NVIDIA dùng để tối ưu Deep Learning inference trên GPU NVIDIA.

Pipeline:

```text
PyTorch / TensorFlow
        ↓
      ONNX
        ↓
    TensorRT
        ↓
Optimized Engine
        ↓
Inference
```

TensorRT hỗ trợ các kỹ thuật tối ưu như:

- FP16
- INT8
- Layer Fusion
- Kernel Optimization

Đặc biệt phù hợp với:

- NVIDIA Jetson
- Edge GPU
- Robotics
- Computer Vision
- ADAS

---

# 43. Deep Learning trong ADAS

Deep Learning đóng vai trò quan trọng trong các hệ thống ADAS.

Ví dụ:

```text
Camera
 ↓
Object Detection
 ↓
Vehicle / Pedestrian / Traffic Sign
 ↓
Object Tracking
 ↓
Decision
 ↓
ADAS Control
```

Các bài toán:

- Lane Detection
- Vehicle Detection
- Pedestrian Detection
- Traffic Sign Recognition
- Driver Monitoring
- Collision Detection
- Object Tracking

---

# 44. Ví dụ kiến trúc ADAS

```text
Camera
   │
   ↓
Image Preprocessing
   │
   ↓
YOLO / CNN
   │
   ↓
Object Detection
   │
   ├── Vehicle
   ├── Pedestrian
   ├── Traffic Sign
   └── Lane
   │
   ↓
Decision System
   │
   ↓
ADAS Controller
   │
   ├── Warning
   ├── Braking
   └── Steering Assistance
```

---

# 45. Deep Learning Pipeline

Một pipeline Deep Learning hoàn chỉnh:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Data Preprocessing
      ↓
Train / Validation / Test
      ↓
Model Architecture
      ↓
Training
      ↓
Evaluation
      ↓
Hyperparameter Tuning
      ↓
Model Optimization
      ↓
Deployment
      ↓
Inference
      ↓
Monitoring
```

---

# 46. Framework phổ biến

Một số framework Deep Learning:

### PyTorch

Phù hợp cho:

- Research
- Deep Learning
- Computer Vision
- NLP
- Model Development

### TensorFlow

Phù hợp cho:

- Deep Learning
- Production
- Deployment
- TensorFlow Lite

### Keras

API cấp cao giúp xây dựng Neural Network nhanh và đơn giản.

---

# 47. Hardware cho Deep Learning

Training Deep Learning thường sử dụng:

```text
CPU
 ↓
GPU
 ↓
TPU
```

GPU đặc biệt phù hợp vì Deep Learning cần thực hiện rất nhiều phép toán song song.

---

# 48. CPU vs GPU

| CPU | GPU |
|---|---|
| Ít core hơn | Rất nhiều CUDA cores/compute units |
| Tốt cho xử lý tuần tự | Tốt cho xử lý song song |
| Phù hợp General Purpose | Phù hợp Matrix Computation |
| Training DL thường chậm hơn | Training DL thường nhanh hơn |

---

# 49. Một số thuật ngữ cần nhớ

| Thuật ngữ | Ý nghĩa |
|---|---|
| Neural Network | Mạng nơ-ron |
| Neuron | Đơn vị tính toán |
| Weight | Trọng số |
| Bias | Độ lệch |
| Parameter | Weight + Bias |
| Hyperparameter | Tham số được thiết lập trước training |
| Epoch | Một lần đi qua toàn bộ dataset |
| Batch | Nhóm sample |
| Learning Rate | Tốc độ học |
| Loss | Mức độ sai |
| Gradient | Hướng thay đổi Loss |
| Backpropagation | Lan truyền ngược |
| Optimizer | Bộ tối ưu |
| Inference | Quá trình dự đoán |
| Training | Quá trình học |
| CNN | Neural Network cho Vision |
| RNN | Neural Network cho Sequence |
| LSTM | RNN có bộ nhớ dài hạn |
| GRU | Biến thể RNN |
| Transformer | Kiến trúc dựa trên Attention |
| Quantization | Giảm precision |
| Pruning | Loại bỏ parameter |
| TinyML | ML trên thiết bị nhỏ |
| Edge AI | AI tại Edge Device |

---

# 50. Tổng kết

Deep Learning có thể được hiểu theo chuỗi:

```text
Data
 ↓
Neural Network
 ↓
Forward Propagation
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradient
 ↓
Optimizer
 ↓
Update Weights
 ↓
Training
 ↓
Evaluation
 ↓
Deployment
 ↓
Inference
```

Các kiến trúc quan trọng:

```text
Deep Learning
│
├── Neural Network
│
├── CNN
│   └── Computer Vision
│
├── RNN
│   ├── LSTM
│   └── GRU
│       └── Time Series / Sequence
│
└── Transformer
    ├── NLP
    ├── Computer Vision
    └── Multimodal AI
```

Đối với định hướng **Embedded AI / Edge AI / AIoT / ADAS**, có thể đi theo:

```text
Machine Learning
        ↓
Deep Learning
        ↓
Neural Network
        ↓
CNN / RNN / Transformer
        ↓
TinyML
        ↓
Model Optimization
        ↓
Edge AI
        ↓
AIoT
        ↓
ADAS
```

> **Deep Learning là phương pháp sử dụng các mạng Neural Network nhiều tầng để tự động học các đặc trưng và mối quan hệ phức tạp từ dữ liệu.**

## 🎯 Kiến thức cần nắm chắc

Để học Deep Learning tốt, cần nắm theo thứ tự:

```text
1. Neural Network
        ↓
2. Neuron / Weight / Bias
        ↓
3. Activation Function
        ↓
4. Forward Propagation
        ↓
5. Loss Function
        ↓
6. Backpropagation
        ↓
7. Gradient Descent
        ↓
8. Optimizer
        ↓
9. CNN
        ↓
10. RNN / LSTM / GRU
        ↓
11. Transformer
        ↓
12. Model Optimization
        ↓
13. TinyML
        ↓
14. Edge AI
```
````

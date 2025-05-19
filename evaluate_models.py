import json
import numpy as np
import matplotlib.pyplot as plt

# 1. Đường dẫn tới các file history
models = {
    'VGG16':          'vgg16_4.json',
    'ResNet50':       'resnet50_leaf_disease.json',
    'MobileNetV2':    'mobilenetv2_leaf_disease.json',
}

# Các tên khóa khả dĩ cho accuracy và loss
acc_keys  = ['val_accuracy', 'val_acc', 'validation_accuracy']
loss_keys = ['val_loss', 'validation_loss']

best_acc  = {}
best_loss = {}

for name, path in models.items():
    with open(path, 'r') as f:
        history = json.load(f)
    # In ra các khóa để debug
    print(f"\n{name} keys:", list(history.keys()))

    # Tìm khóa accuracy và loss
    key_acc = next((k for k in acc_keys if k in history), None)
    key_loss = next((k for k in loss_keys if k in history), None)

    if key_acc is None or key_loss is None:
        raise ValueError(
            f"[{name}] Không tìm thấy khóa accuracy ({acc_keys}) hoặc loss ({loss_keys}) trong {path}."
        )

    val_acc_list  = history[key_acc]
    val_loss_list = history[key_loss]

    if not isinstance(val_acc_list, list) or not isinstance(val_loss_list, list):
        raise ValueError(f"[{name}] Dữ liệu {key_acc}/{key_loss} phải là list trong {path}.")

    # Tìm epoch có val_accuracy cao nhất
    idx_best = int(np.argmax(val_acc_list))
    best_acc[name]  = val_acc_list[idx_best]
    best_loss[name] = val_loss_list[idx_best]
    print(f"{name}: best {key_acc} = {best_acc[name]:.4f} @ epoch {idx_best+1}, {key_loss} = {best_loss[name]:.4f}")

# 3. Chuẩn bị dữ liệu cho biểu đồ
labels     = list(models.keys())
acc_values = [best_acc[m]   for m in labels]
loss_values= [best_loss[m]  for m in labels]

x = np.arange(len(labels))
width = 0.35

# 4. Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, acc_values,  width, label='Best Val Accuracy')
bars2 = ax.bar(x + width/2, loss_values, width, label='Val Loss cùng epoch')

ax.set_xlabel('Mô hình')
ax.set_ylabel('Giá trị')
ax.set_title('So sánh hiệu năng trên tập validation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(bars):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.show()

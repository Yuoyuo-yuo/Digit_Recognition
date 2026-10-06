import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# 1. 加载数据：X 是图片的像素，y 是图片对应的数字
digits = load_digits()
X = digits.data
y = digits.target

print("图片数量：", len(X))
print("每张图片的尺寸：", digits.images[0].shape)

# 2. 分出训练集和测试集
# 训练集用于学习；测试集用于检查对未见过图片的识别效果
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=36, stratify=y
)

# 3. 训练模型
model = SVC(gamma=0.001)
model.fit(X_train, y_train)

# 4. 识别测试集中的图片
predictions = model.predict(X_test)
print(f"识别准确率：{accuracy_score(y_test, predictions):.2%}")

# 5. 展示图片、真实数字和预测结果
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
    ax.set_title(f"True: {y_test[i]} | Pred: {predictions[i]}")
    ax.axis("off")
plt.tight_layout()

# 6. 分析哪些数字容易被识别错
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Digit Recognition Results")

# 找出预测结果与真实数字不同的图片
wrong_indices = [
    i for i in range(len(y_test))
    if predictions[i] != y_test[i]
]

print("测试图片数量：", len(y_test))
print("识别错误数量：", len(wrong_indices))

# 展示所有识别错误的图片
if wrong_indices:
    fig, axes = plt.subplots(
        1, len(wrong_indices),
        figsize=(3 * len(wrong_indices), 3),
        squeeze=False
    )

    for ax, i in zip(axes.flat, wrong_indices):
        ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
        ax.set_title(f"True: {y_test[i]} | Pred: {predictions[i]}")
        ax.axis("off")

    plt.tight_layout()

plt.show()
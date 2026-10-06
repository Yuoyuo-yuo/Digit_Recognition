# 手写数字识别

使用 Python 和 scikit-learn 的 SVC 分类器识别手写数字 0～9。

## 安装与运行

安装依赖：

    python -m pip install scikit-learn matplotlib

运行程序：

    python digit_recognition.py

## 项目功能

- 展示图片的真实标签和预测结果
- 计算识别准确率
- 绘制混淆矩阵
- 展示识别错误的图片

## 数据与模型

- 数据集：digits，共 1797 张 8×8 灰度图片
- 数据划分：80% 训练，20% 测试
- 随机种子：random_state=36
- 模型：SVC(gamma=0.001)

## 测试结果

360 张测试图片中，正确识别 357 张，准确率为 **99.17%**。

3 张错误分别为：9 → 8、5 → 6、5 → 9。

本次测试集结果如下:
<img width="986" height="330" alt="屏幕截图 2026-10-06 162003" src="https://github.com/user-attachments/assets/ad6f34c3-3c24-409f-803e-d3aac1d28a9a" />
<img width="871" height="656" alt="屏幕截图 2026-10-06 162000" src="https://github.com/user-attachments/assets/6b11e830-81fc-4196-a280-f73fe276d240" />
<img width="992" height="408" alt="屏幕截图 2026-10-06 161957" src="https://github.com/user-attachments/assets/ef32868d-7589-4974-afea-2dc621cd7064" />
<img width="270" height="229" alt="屏幕截图 2026-10-06 162006" src="https://github.com/user-attachments/assets/a6ceae32-61d0-4bd2-bcb8-824da973801e" />


## 参考

[scikit-learn 官方示例](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)

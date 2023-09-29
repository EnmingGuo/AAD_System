import numpy as np
import matplotlib.pyplot as plt

# 随机生成数据
np.random.seed(42)
n_samples = 180

# 类别1的数据 - Evolving GCN
scores_evolving = np.random.normal(loc=0.75, scale=0.08, size=n_samples)
labels_evolving = np.ones(n_samples)

# 类别0的数据 - 静态 GCN
scores_static = np.random.normal(loc=0.65, scale=0.08, size=n_samples)
labels_static = np.zeros(n_samples)

# 合并数据和标签
scores = np.concatenate([scores_evolving, scores_static])
labels = np.concatenate([labels_evolving, labels_static])

# 计算每个类别的TPR和FPR
sorted_indices = np.argsort(scores)
labels = np.array(labels)[sorted_indices]
TPR = np.cumsum(labels) / np.sum(labels)
FPR = np.cumsum(~labels) / np.sum(~labels)

# 添加起伏感
noise = np.random.normal(loc=0, scale=0.02, size=len(TPR))
TPR += noise
FPR += noise

# 绘制ROC曲线
plt.plot(FPR, TPR, 'b-', linewidth=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')

# 添加对角线（随机分类器）的参考线
plt.plot([0, 1], [0, 1], 'r--')

# 显示图形
plt.show()

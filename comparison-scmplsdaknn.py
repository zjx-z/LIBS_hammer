import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_decomposition import PLSRegression
from joblib import dump, load

# 设置随机数种子
np.random.seed(42)


# 加载训练集和测试集
train_data = pd.read_csv('E:\\libs\\data1245-50.csv')  # 替换为你的训练集文件路径
test_data = pd.read_csv('E:\\libs\\data36-50.csv')    # 替换为你的测试集文件路径

# 假设最后一列是目标变量，其余为特征
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

algorithm = "RANDOM_FOREST"
if algorithm == "PCA-SVM":
    # PCA降维
    pca = PCA(n_components=200)  # 选择保留的主成分数

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 输出PCA的总贡献率
    explained_variance_ratio = pca.explained_variance_ratio_
    total_explained_variance_ratio = np.sum(explained_variance_ratio)
    print(f"PCA Explained Variance Ratio for each component: {explained_variance_ratio}")
    print(f"Total Explained Variance Ratio (PCA): {total_explained_variance_ratio:.4f}")

    # SVM分类器
    svm = SVC(kernel='linear')
    svm.fit(X_train_pca, y_train)

    # 预测与评估
    y_pred_pca_svm = svm.predict(X_test_pca)
    print("PCA-SVM Accuracy:", accuracy_score(y_test, y_pred_pca_svm))
    print("PCA-SVM Classification Report:\n", classification_report(y_test, y_pred_pca_svm))

    data = pd.read_csv("E:\libs\data143.csv")
    X_rank = data.iloc[:, :].values
    y_pred = svm.predict(X_rank)
    num=["1-1","1-10","1-20","43-1"]
    t=0
    for y_pred1 in y_pred:
        #print(num[t])
        t+=1
        # 对列表进行降序排序，并获取索引
        sorted_enumerate = sorted(enumerate(y_pred1), key=lambda x: x[1], reverse=True)
        # 按照索引：值的格式输出
        for index, value in sorted_enumerate:
            print(f"{index}: {value}")

elif algorithm == "SVM":
    # 直接使用SVM
    svm = SVC(kernel='rbf', probability=True, random_state=42)#SVM Accuracy: 0.9811320754716981
    svm.fit(X_train, y_train)

    # 预测与评估
    y_pred_svm = svm.predict(X_test)
    #print("y_pred_svm")
    #print(y_pred_svm)
    y_pred_svm_proba=svm.predict_proba(X_test)
    print("y_pred_svm_proba")
    print(y_pred_svm_proba)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

    data = pd.read_csv("E:\libs\data143.csv")
    # 获取类别标签
    classes = svm.classes_
    X_rank = data.iloc[:, :].values
    y_rank=svm.predict(X_rank)
    #print("y_rank")
    #print(y_rank)
    num=["1-1","1-10","1-20","43-1"]
    # 获取新数据的预测概率
    y_prob = svm.predict_proba(X_rank)
    # 打印每个样本的预测概率
    num = ["1-1", "1-10", "1-20", "43-1","103-1"]
    t = 0
    for proba in y_prob:
        #print(proba)
        print(f"{num[t]}: Probabilities - ", end='')
        for i, class_prob in enumerate(proba):
            print(f"Class {classes[i]}: {class_prob:.4f}", end='  ')
        print()  # 换行
        t += 1
    dump(svm, 'svm_model.joblib')
elif algorithm == "PLS-DA":
    # PLS-DA
    pls_da = PLSRegression(n_components=200)  # 选择保留的成分数
    pls_da.fit(X_train, y_train)

    # 预测与评估
    y_pred_pls_da = pls_da.predict(X_test)
    y_pred_pls_da = np.round(y_pred_pls_da).flatten()  # 将连续预测值四舍五入为类别标签
    print("PLS-DA Accuracy:", accuracy_score(y_test, y_pred_pls_da))
    print("PLS-DA Classification Report:\n", classification_report(y_test, y_pred_pls_da))
elif algorithm == "PCA-KNN":
    # PCA降维
    pca = PCA(n_components=200)  # 选择保留的主成分数
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # KNN分类器
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)

    # 预测与评估
    y_pred_pca_knn = knn.predict(X_test_pca)
    print("PCA-KNN Accuracy:", accuracy_score(y_test, y_pred_pca_knn))
    print("PCA-KNN Classification Report:\n", classification_report(y_test, y_pred_pca_knn))

    data = pd.read_csv("E:\libs\data143.csv")
    X_rank = data.iloc[:, :].values
    y_pred = knn.predict(X_rank)
    num=["1-1","1-10","1-20","43-1"]
    t=0
    for y_pred1 in y_pred:
        print(num[t])
        t+=1
        # 对列表进行降序排序，并获取索引
        sorted_enumerate = sorted(enumerate(y_pred1), key=lambda x: x[1], reverse=True)
        # 按照索引：值的格式输出
        for index, value in sorted_enumerate:
            print(f"{index}: {value}")

elif algorithm == "KNN":
    # 直接使用KNN
    knn = KNeighborsClassifier(n_neighbors=12)#KNN Accuracy: 0.9811320754716981
    knn.fit(X_train, y_train)

    # 预测与评估
    y_pred_knn = knn.predict(X_test)
    print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
    print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
    data = pd.read_csv("E:\libs\data143.csv")
    # 获取类别标签
    classes = knn.classes_
    X_rank = data.iloc[:, :].values
    y_rank = knn.predict(X_rank)
    # print("y_rank")
    # print(y_rank)
    num = ["1-1", "1-10", "1-20", "43-1"]
    # 获取新数据的预测概率
    y_prob = knn.predict_proba(X_rank)
    # 打印每个样本的预测概率
    num = ["1-1", "1-10", "1-20", "43-1","103-1"]
    t = 0
    for proba in y_prob:
        # print(proba)
        print(f"{num[t]}: Probabilities - ", end='')
        for i, class_prob in enumerate(proba):
            print(f"Class {classes[i]}: {class_prob:.4f}", end='  ')
        print()  # 换行
        t += 1
    dump(knn, 'knn_model.joblib')
elif algorithm == "RANDOM_FOREST":
    # 随机森林分类器
    rf = RandomForestClassifier(n_estimators=100, random_state=42)#Random Forest Accuracy: 0.9784366576819407
    rf.fit(X_train, y_train)

    # 预测与评估
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
    data = pd.read_csv("E:\libs\data143.csv")
    # 获取类别标签
    classes = rf.classes_
    X_rank = data.iloc[:, :].values
    y_rank = rf.predict(X_rank)
    # print("y_rank")
    # print(y_rank)
    num = ["1-1", "1-10", "1-20", "43-1"]
    # 获取新数据的预测概率
    y_prob = rf.predict_proba(X_rank)
    # 打印每个样本的预测概率
    num = ["1-1", "1-10", "1-20", "43-1","103-1"]
    t = 0
    for proba in y_prob:
        # print(proba)
        print(f"{num[t]}: Probabilities - ", end='')
        for i, class_prob in enumerate(proba):
            print(f"Class {classes[i]}: {class_prob:.4f}", end='  ')
        print()  # 换行
        t += 1
    dump(rf, 'rf_model.joblib')

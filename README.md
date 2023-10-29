# Titanic - Machine Learning from Disaster
Start here! Predict survival on the Titanic and get familiar with ML basics  
kaggle: https://www.kaggle.com/c/titanic
預測鐵達尼號乘客生存情形

-------

## 成績 (2022.02時点)

 * Test Acc : 0.79665
 * Top 5% (665/13,847)

-------

## 内容

### EDA
 * 生存状況分布
 * 相関係数相関係数
 * 欠損値状況
 * 数値データ分布

### 特徴量エンジニアリング
 * 特徴量作り： 
    * 性別　→　ダミー変数 （男性=1、女性=0 ）
    * 年齢　→　グループに分けて、ダミー変数にする
    * 新特徴量：名前の長さ（単語数）
    * 新特徴：身分を表する単語（Mr., Mrsなど）
 * One-Hot Encoding (年齡、Pclass、Embarked) 
 * 欠損値を埋める
  * 欠損率高すぎの特徴 => 削除
  * カテゴリ、年齡 => 'none'
  * Fare => 平均

### 特徴量標準化、データサンプリング
 * データサンプリング: train set => 85%, valid set => 15%
 * PassengerId, Survived 以外の特徴量を標準化（train setを基に）

### モデル構築、予測 (Grid search でハイパーパラメータを調整)
 * Logistic
 * KNN
 * SVM
 * Decision Tree
 * Random Forest 

### ブレンド方法
 * Voting: 上記５つモデルの予測で多数決 (VotingClassifier)
 * Stacking: 上記５つモデルの予測を変数として、LogisticRegressionモデルを構築 (StackingClassifier)

### Neural Network Model (PyTorch)
 * Model:　Linear + Dropout + ReLU + sigmoid
 * optimizer: (1)SGD + momentum + weight_decay、(2)Adam、(3)AdamW
 * Voting: 上記３つoptimizerで作ったモデルの予測で多数決

-------

## 内容

### EDA
 * 乘客存活比例、男女/年齡分布
 * 存活情形與其他變數的相關性
 * 遺漏值確認
 * 數值類變數的分布情形

### 特徵工程
 * 製造特徵: 
    * 性別 => 男性為1、女性為0 
    * 姓名單字數
    * 缺值欄位數
    * 年齡 => 各年齡層組別
    * 姓名中的稱謂字 ('Mr.', 'Mrs.', 'Miss.', 'Master.')
 * One-Hot Encoding (年齡組、Pclass、Embarked) 
 * 處理遺漏值
  * 缺值比例過高欄位 => 移除
  * 類別欄位、年齡 => 補'none'
  * Fare => 補訓練資料平均數

### 標準化、切割訓練/測試/驗證集
 * 以訓練資料為基準將 PassengerId, Survived 以外其他欄位標準化
 * 切割: train set => 85%, valid set => 15%

### 單一模型訓練、預測 (Grid search 找最佳參數)
 * Logistic
 * KNN
 * SVM
 * Decision Tree
 * Random Forest 

### 集成模型
 * Voting: 前面的5個模型1人1票 (VotingClassifier)
 * Stacking: 前面的5個模型當成變數 => LogisticRegression (StackingClassifier)

### Neural Network Model (PyTorch)
 * Model:　Linear + Dropout + ReLU + sigmoid
 * optimizer: (1)SGD + momentum + weight_decay、(2)Adam、(3)AdamW
 * Voting: 1模型1票多數決

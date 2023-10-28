# Titanic - Machine Learning from Disaster
Start here! Predict survival on the Titanic and get familiar with ML basics  
kaggle: https://www.kaggle.com/c/titanic
預測鐵達尼號乘客生存情形

## 成績

 * Test Acc : 0.79665
 * Top 5% (665/13,847)


## 内容

1. ### EDA
 * 乘客存活比例、男女/年齡分布
 * 存活情形與其他變數的相關性
 * 欠損データの確認
 * 數值類變數的分布情形

2. ### 特徵工程
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

3. ### 標準化、切割訓練/測試/驗證集
 * 以訓練資料為基準將 PassengerId, Survived 以外其他欄位標準化
 * 切割: train set => 85%, valid set => 15%

4. ### 單一模型訓練、預測 (Grid search 找最佳參數)
 * Logistic
 * KNN
 * SVM
 * Decision Tree
 * Random Forest 

5. ### 集成模型
 * Voting: 前面的5個模型1人1票 (VotingClassifier)
 * Stacking: 前面的5個模型 + LogisticRegression (StackingClassifier)

6. ### Neural Network Model (PyTorch)
 * Model:　Linear + Dropout + ReLU + sigmoid
 * optimizer: SGD + momentum + weight_decay / Adam /AdamW
 * Voting: 1模型1票多數決

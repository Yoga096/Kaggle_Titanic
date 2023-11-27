# Titanic - Machine Learning from Disaster
Start here! Predict survival on the Titanic and get familiar with ML basics  
kaggle: https://www.kaggle.com/c/titanic
預測鐵達尼號乘客生存情形

-------

## 成績 (2022.02時点)

 * Test Acc : 0.79665
 * Top 5% (665/13,847)

-------

## 内容（日本語 ver.）

### EDA
#### カテゴリ変数により生存状況を確認
- 女性、ハイクラスの乗客の生存率が高い
- Cherbourgから乗船した人の生存率が高い
- 家族（兄弟、父母、子供）が一緒にいない人の生存率が低い

#### 年齢・性別について生存状況を確認
- 10代後半から男性の生存率が低い
- 5~15歳の区間の女性生存率が比較的に低い
- 年齢と生存率の相関は単純な直線ではない ⇒ カテゴリ変数にする方が良い

#### 更にPclassを考慮する
- 高いクラスの男性生存率が低いクラスよりも高い
- 低いクラスの女性生存率が高いクラスよりも低い

#### 運賃から見る場合
- Pclassから見るの結果とほぼ同じ

#### 各変数の相関関係
- 相関係数
- ヒートマップで表示


### 特徴量エンジニアリング
#### 欠損値を埋める
- Age: 年齢データなしの乗客の生存率は他の乗客よりも低い ⇒ Age=NAのデータを保留する
- Cabin: 座席データの有無により生存率が異なる ⇒ Cabin=NAのデータを保留する
#### 年齢データをグループに分けて、ダミー変数にする
#### 名前で身分を区分する
- 社会階級は乗客の生存率に影響する
- 乗客の名前から社会階級を判断できる
- 名前の長さ（単語数）
- 身分を表する単語（Mr., Ms., Dr.など）
#### その他
- fillna(Embarked)
- One-Hot Encoding

### モデル構築の準備
#### データサンプリング
- ランダムに15%の訓練データを検証用のvalid dataにする
#### 特徴量の選択・標準化
- 訓練データを基に全ての特徴量を標準化する

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

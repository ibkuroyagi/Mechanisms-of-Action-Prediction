# 1st solution
https://www.kaggle.com/c/lish-moa/discussion/201510
## 1. どんなもの？
- 7モデルのブレンディング
    * 3-stage NN stacking by non-scored and scored meta-features(weight高い)
    * 2-stage NN+TabNet stacking by non-scored meta-features
    * SimpleNN with old CV
    * SimpleNN with new CV
    * 2-heads ResNet
    * DeepInsight EfficientNet B3 NS
    * DeepInsight ResNeSt

## 2. 自分達の手法や流行との相違点
- スタッキングのメタ特徴量を多様している
(n-stage model)
- テストデータの説明変数情報は使っていない
## 3. 技術や手法のキモはどこ？
- 特徴量エンジニアリング
    * Quantile Transformer
    * PCA
    * UMAP
    * Factor Analysis
    * t-test Feature Selection
    * K-means Clustering
    * statistics features
    * Onehot Encoding 
    * Variance Threshold
- MLP, Tabnet
    * 2-stage model(MLP+Tabnet), 3-stage model, old CV, new CVの計4つ
    
    * **3　stage**
        ![3stage](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F201774%2Ffb1fcba2f3331b3d3ac90cda457c23bd%2Ffigure_2.png?generation=1607164173264589&alt=media)
    
    * **2　stage**
        ![2stage](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F201774%2F7cec080edde3393d3c86a779d227b211%2Ffigure_4.png?generation=1607164563571085&alt=media)

    * **1　stage**
        ![1stage](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F201774%2F8276e3b36e68f6ad4557b22ea5749c46%2Ffigure_6.png?generation=1607175455595627&alt=media)

    * 2,3 stageのものはnon scoredをスタッキングの途中で予測させて、Quantile Normをして後続のスタッキングの特徴にしている
    * 3 stage/1 stageのものは特徴選択していない
    * 2層目で、1層目で行ってないFEをしている
    * MLPアーキテクチャ 
        - bn -> dropout -> Dense(2048) -> LeakyReLU(2048)
        - の3層
        * hyper param
            - epoch=25
            - optimizer = Adam
            - lr = 5e-3
            - batch = 256
            - weight decay = 1e-5
            - scheduler = OneCycleLR
                - maxlr = 1e-2
    * Tabnet
        - mask size = 32 x 32
        - n_steps = 1
        - gamma = 0.9 (baseline=1.3)
        - hyper param
            - optimizer = adam
            - lr = 2e-2
            - weight decay = 1e-5
            - mask = entmax
            - epoch=200
            - early stopping=50
- 2 heads ResNet
    * 全特徴+PCAのinput1
    * t-test feature selection + onehot-cateのinput2
    * non-scoredを予測してfinetuneしてscored予測？

- Deepinsight CNNs
    * t-SNEで画像化等の前処理をしてEfficientNet, Resnet
    * control groupは精度向上に貢献した
    - CNN -> Dense(512) -> ELU -> Dense(206) -> sigmoid
## 4. どんな戦略？どんな考察がされていた？
 * weight decay, label smoothingが過学習対策に大事だった
 * CV/LBに効いているモデルを良いモデル
 * DeepInsightは0.63-0.73の相関でスタッキングに役立った
 * Conclusionより(訳)
* * *
    MoAや医療領域に関する予備知識がなくても、私たちはこの小さくてバランスの悪いデータセットに、私たちが知っているML/DL技術を適用するために最善を尽くしました。最終的な立ち位置にはかなり驚かされましたが、一般的なAI手法はほとんどどのような領域でも機能することがわかりました。
    異なるアーキテクチャの浅いNNモデルの組み合わせ、詳細な特徴工学、多段階スタッキング、ラベル平滑化、伝達学習、多様なDeepInsight CNNの追加が、私たちのチームの最終的な提出物の主な勝利要因です。
    この長い記事を読んでいただきありがとうございました。
* * *
## 5. 議論や考察
![blend table](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F201774%2Fd3f48b386b46934c95d534f7007233ba%2Ftable_1.png?generation=1607160513209233&alt=media)
* New CVのスコアが高い。薬品情報を使ったCVは効果的だったことが考えられる

## 6. 次に読むべき論文/notebook/discussionは？
- Factor Analysis(因子分析)って何をしてる？
- Deepinsight Feature map
(https://www.kaggle.com/markpeng/deepinsight-transforming-non-image-data-to-images)
- ELU activation is 何？
- swap noise is 何？

# 全体でどんな手法が使われていた？
- 1D-CNN(2nd solution)
- 追加データでのCVだいじ
- no ctrlを使ったdata augumentation (3rd solution)
- 2D CNN stacking(4th solution)
- combination features(4th solution)
    * g特徴量 + c特徴量を2組で足して、variance thresholdでいらないものを適宜捨てる
- metric learningを用いたseen,unseen分類(5th solution)

# 総括
- ドメイン特有のfeature engineeringは行われてないように見える。
- 基本的な機械学習/深層学習テクニックが問われるコンペだった。
- テーブルデータだったが、NN力が問われるコンペだった。
- （NNの）これからはNNモデリング力がどんどん問われるようになっていくのでは？そんなことを感じさせるコンペだった。
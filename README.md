# Mechanisms of Action (MoA) Prediction
## コンペの概要ページ
[https://www.kaggle.com/c/lish-moa/overview](https://www.kaggle.com/c/lish-moa/overview)

## Final submission deadline.
- November 30, 2020
### 反省会資料
[https://docs.google.com/presentation/d/1BssOEzBWpJ8X8wV5wpbHCWWZ9vAGEPzVn5l34Lu2xBk/edit?usp=sharing](https://docs.google.com/presentation/d/1BssOEzBWpJ8X8wV5wpbHCWWZ9vAGEPzVn5l34Lu2xBk/edit?usp=sharing0)
fkubotaさんのスライドを拝借しました

### 日本語訳
本コンテストでは、遺伝子発現データや細胞生存率データなどの様々なインプットを与えられた異なるサンプル（sig_id）の作用機序（MoA）応答の複数のターゲットを予測

2つの注意点がある

1. トレーニングデータには、テストデータには含まれず、スコアリングには使用されないMoAラベルの追加（オプション）セットがある
2. 再実行データセットは、パブリックテストで見られる例の約4倍の数を持つ
### ファイル
- train_features.csv
    * g-接頭辞を持つ特徴量は遺伝子発現特徴量であり、その数は772個（g-0からg-771まで）ある
    * c-接頭辞を持つ特徴量は細胞生存率の特徴量であり、その数は100個（c-0からc-99まで）ある
            * ただし、g,cともに匿名化されているため具体的にどんな遺伝子や細胞の反応かを知ることはできない
    * cp_typeは，サンプルが化合物で処理されたか，対照摂動（rt_cpまたはctl_vehicle）で処理されたかを示す2値のカテゴリ特徴量
    * cp_timeは，治療期間（24時間，48時間，72時間）を示す分類的特徴量
    * cp_doseは，投与量が低いか高いかを示す2値のカテゴリ特徴量である(D1またはD2)．
- train_targets_scored.csv - スコアされるバイナリMoAターゲット
    * Number of Scored Target Features: 206
- train_targets_nonscored.csv - 訓練データの追加の（オプションの）バイナリMoA反応。これらは予測もスコア化もされない
    * Number of Non-scored Target Features: 402
- test_features.csv - テストデータの特徴量．テストデータの各行のスコアされたMoAの確率を予測する必要がある
- sample_submission.csv - 正しい形式の提出ファイル

train_features.csv
![2020-12-22 (1)](https://user-images.githubusercontent.com/44137906/102840315-ec21c380-4445-11eb-970b-d3ac1bb83ea2.png)
train_targets_scored.csv
![2020-12-22](https://user-images.githubusercontent.com/44137906/102840317-ed52f080-4445-11eb-9f2b-2aa9eb07f66f.png)


## 評価指標
各カラムにおけるバイナリクロスエントロピーを計算しその平均値を、すべてのサンプルで平均した値
![2020-12-22 (2)](https://user-images.githubusercontent.com/44137906/102840475-520e4b00-4446-11eb-8d2a-dd478ed09beb.png)

## お願い
* multistratifiledkfoldを使う
    * sys.path.append('../input/iterative-stratification/iterative-stratification-master')
    *  kfold = MultilabelStratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=True)
    * [https://www.kaggle.com/gogo827jz/moa-public-pytorch-node](https://www.kaggle.com/gogo827jz/moa-public-pytorch-node)
- Out of Fold(oof)なファイルを必ず作成
- みんながsubmitしやすいようなnotebookを作成

## 実験結果
実験結果はこのシートに記載
[https://docs.google.com/spreadsheets/d/1eXo1StHFrhRA3AXmJbf_wdKSmB9vPXuRcrv32GBCi-8/edit#gid=0](https://docs.google.com/spreadsheets/d/1eXo1StHFrhRA3AXmJbf_wdKSmB9vPXuRcrv32GBCi-8/edit#gid=0)

## ToDo
- [https://www.kaggle.com/c/lish-moa/discussion/184005](https://www.kaggle.com/c/lish-moa/discussion/184005)
* データの前処理
- [x] ~~label smooth~~
- [x] ~~トレーニングデータからコントロールグループを削除するとCV上がるらしい.LBは下がるけどブレンドするとLBもup~~
- [ ] pretictのlowerとupperをclip
- [x] ~~コントロールの出力を全て確認。もしかしたらすべて0かも~~
- [x] ~~Neural Oblivious Decision Ensembles(NNで作った多段決定木をredualに結合して出力)~~
- [x] ~~Pseudolabeling/noisy label training~~
- [ ] バランシングの適応 (優先順位低め)
- [ ] アップサンプリング[https://www.kaggle.com/c/lish-moa/discussion/187419](https://www.kaggle.com/c/lish-moa/discussion/187419)
    * [ノートブック](https://www.kaggle.com/tolgadincer/upsampling-multilabel-data-with-mlsmote)
    * [CVの方法のノートブック](https://www.kaggle.com/tolgadincer/mlsmote) (優先順位低め)
- [ ] ImbalancedDataSampler[pytorchの実装github](https://github.com/ufoym/imbalanced-dataset-sampler)
* 特徴量作成
* 成田タスク
    - [x] ~~ノンスコアのターゲットを予測し、その後のモデルのメタ特徴として使用~~
    - [x] ~~ノンスコアのターゲットも含めたモデルで学習する~~
    - [x] ~~TABNetの重要度を使用する~~
* 畔栁タスク
    - [x] ~~TabNetベースライン作成~~
    - [x] ~~PCA~~
    - [x] ~~DPGMM~~
    - [x] ~~Z-score~~
    - [x] ~~RankGauss~~
    - [x] ~~noisy label training~~
    - [x] ~~globalな特徴量作成(testデータの情報を追加)~~
    - [ ] AEでデノイズor中間層を特徴量に追加
    - [ ] メトリックラーニングをAEに適応して、他のモデルの特徴量にする (クラスタリングの重心を特徴量に加えるイメージ)
    - [x] ~~CNNベースのスタッキング~~
    * モデル構造
    - [ ] Tabunetの中間層でグループを作成すると,全結合のみよりもCVがgood
    - [ ] split NN keras[https://www.kaggle.com/gogo827jz/split-neural-network-approach-tf-keras](https://www.kaggle.com/gogo827jz/split-neural-network-approach-tf-keras)
    - [ ] TabNet num_decision_steps = 1 makes OOF score much better [https://www.kaggle.com/gogo827jz/moa-stacked-tabnet-baseline-tensorflow-2-0](https://www.kaggle.com/gogo827jz/moa-stacked-tabnet-baseline-tensorflow-2-0)
* 西山タスク
    - [x] ~~SVM~~
    - [x] ~~lgbm~~
    - [ ] xgb
    - [x] ~~kernel Ridge [https://www.kaggle.com/gogo827jz/kernel-logistic-regression-one-for-206-targets](https://www.kaggle.com/gogo827jz/kernel-logistic-regression-one-for-206-targets)~~
    - [x] ~~label smooth~~
    - [x] ~~計算量軽めモデルのハイパラチューニング[すごい重要らしい](https://www.kaggle.com/c/lish-moa/discussion/180918#1000976)~~
    - [ ] label powerset (マルチラベルタスクをマルチクラスタスクに変換するらしい. これを使えばSVMやlgbも一つのモデルでOKなのでは?)[http://scikit.ml/api/skmultilearn.problem_transform.lp.html](http://scikit.ml/api/skmultilearn.problem_transform.lp.html)~~
    - [ ] トレーニングデータからコントロールグループを削除するとCV上がるらしい.LBは下がるけどブレンドするとLBもup
    - [x] ~~ブレンドの割合をminmaxを使って計算~~

## introduction
ホームディレクトリに.kaggleディレクトリが作成されている前提で作成します。
ない場合は、こちら[https://www.currypurin.com/entry/2018/kaggle-api](https://www.currypurin.com/entry/2018/kaggle-api)を参照してください。
```
# リポジトリのクローン
git clone https://github.com/ibkuroyagi/Mechanisms-of-Action-Prediction.git
# 仮想環境の構築
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
# 入力データのダウンロード
mkdir input
cd input
kaggle competitions download -c lish-moa
unzip lish-moa
kaggle datasets download -d yasufuminakama/iterative-stratification
unzip iterative-stratification.zip
rm lish-moa.zip iterative-stratification.zip
```
メモ
- コンペの説明[https://www.kaggle.com/c/lish-moa/discussion/184005](https://www.kaggle.com/c/lish-moa/discussion/184005)
    * トレーニングデータの中には、すべての列が0となるデータが40%ある（nonscoreを含めると23%）
    * ブレンドが効く
    * データセット全体では、約5000種類のユニークな薬剤が含まれる
        * 各タイミングx用量レベルで測定されているので同一の薬剤が（異なるIDで）少なくとも6回入っている
    * Dosage(投薬量)が多いほど影響は大きいと考えられる
    * cp_typeのctl_vehicleはMoAを抑制する働きのある投薬をしているので、その影響を排除する前処理（負の値を引き算するだけでも上手くいくかも）
- カエルさん[https://www.kaggle.com/c/lish-moa/discussion/181040](https://www.kaggle.com/c/lish-moa/discussion/181040)
    * trt_cp' は少なくとも一つのラベルを持つ
    * cutout,mixupはhmmmな結果
    * clusteringアプローチもhummm
    * いくつかのラベルは相関があるが共存できない
        * ラベル間の依存性をモデル化するために, MLを使用した後の結果を使用できるか?
-  猫さん[https://www.kaggle.com/c/lish-moa/discussion/183377](https://www.kaggle.com/c/lish-moa/discussion/183377)
    * ブレンドのときのウェイト決定に使えるscipyのコード[https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0](https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0)
    * kernel Ridge
    * Adversarial Validationそんなにいらない
    * PCAは非NNでそれほど効かない
    * マルチラベル分類はNNでしかやっていない、つまり、アルゴリズムベース(SVC, XGB and LGBM)では1target1modelでやっている.そのブレンドは上のコード  
- 不均衡データに対するソリューション一覧 [https://www.kaggle.com/c/lish-moa/discussion/191545](https://www.kaggle.com/c/lish-moa/discussion/191545)
    * アップサンプリング[https://www.kaggle.com/c/lish-moa/discussion/187419](https://www.kaggle.com/c/lish-moa/discussion/187419)
        * [ノートブック](https://www.kaggle.com/tolgadincer/upsampling-multilabel-data-with-mlsmote)
        * [CVの方法のノートブック](https://www.kaggle.com/tolgadincer/mlsmote)
    * ImbalancedDataSampler[pytorchの実装github](https://github.com/ufoym/imbalanced-dataset-sampler)
    * One-shot learning(これなに？)[https://paperswithcode.com/task/one-shot-learning](https://paperswithcode.com/task/one-shot-learning)
    * few shot leaning (これなに？)[https://arxiv.org/pdf/1904.05046.pdf])(https://arxiv.org/pdf/1904.05046.pdf)

* MoAラベルに階層構造を追加する（これやりたい！）
    * 例えば、与えられた薬剤が "拮抗薬 "対 "他の何か "であるかどうかを最初に予測し、次に "拮抗薬 "のクラス内でどの遺伝子セットが影響を受けているかなどを予測することができます。

## 結果
2166/4384 (962 shake down...)  
## 作成したモデル
- TabNet(CV: 0.016978)
    - PCA/statistic/rank-gauss/k-means
    - TabNetの学習済みモデルによる特徴量選択
- NODE(CV: 0.016805)
- MLP(CV: 0.012405)
    - PCA/statistic/rank-gauss/k-means/dpgmm/Variance threshold
    - pseudo-label(0.95以上のサンプルを追加)
    - foldベースのファインチューニング
- StackingCNN (conv1d kernel_size=3) (CV: 0.013989)
- ブレンディング(CV: 0.011401)
```
CV: 0.011401
Public: 0.01890
Private: 0.01669
```
## 敗因
CVの切り方を間違えた。
本来はtestとvalidの分布が一致するべきだが、validをtestによせすぎるとtrainにtestに近いデータがなくなり学習が進まなくなることを嫌ったせいでCVが信用ならない値を取るようになってしまった。
また、fine-tuneの際に1fold前を初期値としたことで完全にvalidがリークを起こしてしまった。
train/testに分布の差がないという前提をadversarial validationで確認したが、さすがに過学習をさせすぎてしまい、そこが致命的なミスとなった。  

また、privateのデータを用いることに固執したことも良くなかった。
それよりもseedを変えて汎か性能を高める努力をするべき(推論のみのコードにする)

loglossに最適化してしまう問題に対処できなかった<- どういったターゲットがただ0に近い値を取るだけになるのかを気づけなかったので、それに応じた対応をとることが困難になった。

<details><summary>kaggle日記</summary><div>

- 10/14(水)
    - 今日やったこと
        * チームアップのミーティング！
        * リポジトリ作成&コンペの理解
    - 次回やること
        * NODEの実行&推論結果を作成
        * ToDo埋める
- 10/15(木)
    - 今日やったこと
        * NODEの実行&推論結果を作成
    - 次回やること
        * kernelにモデルをアップロードしてsubmitを成功させる
- 10/19(月)
    - 今日やったこと
        * kernelにモデルをアップロードしてsubmitを成功させる
    - 次回やること
        * ディスカッションのサーベイ
        - 定期ミーティング
- 10/19(月)
    - 今日やったこと
        * ディスカッションのサーベイ
        - 定期ミーティング -> 担当割り振り完了！
    - 次回やること
        * ベースラインの改造
        * スタッキングノートブック作成方法を模索
- 10/24(土)
    - 今日やったこと
        * ベースラインの改造
    - 次回やること
        * ベースラインによる提出 (5fold NODE QHAdam model) 
- 10/25(日)
    - 今日やったこと
        * ベースラインの改造
    - 次回やること
        * 推論コード作成
- 10/27(火)
    - 今日やったこと
        * 定例ミーティング
        * 推論コード作成&QHAdamのパラメータ調整v000
    - 次回やること
        * 推論コード完成
- 10/28(水)
    - 今日やったこと
        * 推論コード完成
    - 次回やること
        * ノートブックから提出
- 10/28(水)
    - 今日やったこと
        * 推論コード完成
    - 次回やること
        * ノートブックから提出
- 10/29(木)
    - 今日やったこと
        * ノートブックから提出
        * LabelSmoothLoss実装
    - 次回やること
        * LabelSmoothLossノートブックから提出
- 10/30(金)
    - 今日やったこと
        * LabelSmoothLossノートブックから提出
    - 次回やること
        * 前処理のパイプライン作成
- 10/31(土)
    - 今日やったこと
        * 前処理のパイプライン作成
        * LabelSmoothLossパラメータサーチ
    - 次回やること
        * 前処理のパイプラインを用いて適当な前処理探索
- 11/1(日)
    - 今日やったこと
        * 前処理のパイプラインバグ取り
        * dpgmm実装
    - 次回やること
        * v007~v009提出
- 11/2(月)
    - 今日やったこと
        * v007提出
        * v008,v009計算, v007拡張
    - 次回やること
        * v008,v009提出
- 11/3(火)
    - 今日やったこと
        * v007-4,v008,v009提出
        * 定期ミーティング -> on the flyの重要性を確認 
    - 次回やること
        * v007-1~3提出
- 11/4(水)
    - 今日やったこと
        * v007-1~3提出
        * v008-*,v009-*計算
    - 次回やること
        * v008-*,v009-*提出
- 11/16(月)
    - 今日やったこと
        * globalな前処理実装提出
    - 次回やること
        * globalな前処理パイプライン実装
- 11/17(火)
    - 今日やったこと
        * globalな前処理パイプライン実装
    - 次回やること
        * globalな前処理パイプライン実装提出
- 11/18(水)
    - 今日やったこと
        * globalな前処理パイプライン提出v009,v010
        * 定例ミーティング
    - 次回やること
        * noisy label training実装
- 11/19(木)
    - 今日やったこと
        * noisy label training実装&提出v009,v010
    - 次回やること
        * TabNet実装
- 11/25(水)
    - 今日やったこと
        * NODEパラメータサーチ
    - 次回やること
        * fine-tune実装
- 11/25(水)
    - 今日やったこと
        * fine-tune実装
    - 次回やること
        * pseudo-label実装
- 11/26(木)
    - 今日やったこと
        * pseudo-label実装
    - 次回やること
        * CNNスタッキングコード実装
- 11/27(金)
    - 今日やったこと
        * CNNスタッキングコード実装
    - 次回やること
        * CNNスタッキングコード成田くんモデル追加
- 11/28(土)
    - 今日やったこと
        * CNNスタッキングコード成田くんモデル追加
    - 次回やること
        * 提出
- 11/29(日)
    - 今日やったこと
        * CNNスタッキングコード
    - 次回やること
        * ブレンディング追加
- 11/30(月)
    - 今日やったこと
        * ブレンディング追加, verboseを引数処理にして高速化
    - 次回やること
        * 祈る
- 12/1(火)
    - 今日やったこと
        * 祈る
</div></details>

# Mechanisms of Action (MoA) Prediction
## Final submission deadline.
- November 30, 2020

### 日本語訳
本コンテストでは、遺伝子発現データや細胞生存率データなどの様々なインプットを与えられた異なるサンプル（sig_id）の作用機序（MoA）応答の複数のターゲットを予測

2つの注意点がある

トレーニングデータには、テストデータには含まれず、スコアリングには使用されないMoAラベルの追加（オプション）セットがある
再実行データセットは、パブリックテストで見られる例の約4倍の数を持つ
### ファイル
- train_features.csv
    * g-接頭辞を持つ特徴量は遺伝子発現特徴量であり、その数は772個（g-0からg-771まで）ある
    * c-接頭辞を持つ特徴量は細胞生存率の特徴量であり、その数は100個（c-0からc-99まで）ある
    * cp_typeは，サンプルが化合物で処理されたか，対照摂動（rt_cpまたはctl_vehicle）で処理されたかを示す2値のカテゴリ特徴量
    * cp_timeは，治療期間（24時間，48時間，72時間）を示す分類的特徴量
    * cp_doseは，投与量が低いか高いかを示す2値のカテゴリ特徴量である(D1またはD2)．
- train_targets_scored.csv - スコアされるバイナリMoAターゲット
    * Number of Scored Target Features: 206
- train_targets_nonscored.csv - 訓練データの追加の（オプションの）バイナリMoA反応。これらは予測もスコア化もされない
    * Number of Non-scored Target Features: 402
- test_features.csv - テストデータの特徴量．テストデータの各行のスコアされたMoAの確率を予測する必要がある
- sample_submission.csv - 正しい形式の提出ファイル

## 評価指標
各カラムにおけるバイナリクロスエントロピーを計算しその平均値を、すべてのサンプルで平均した値

## お願い
* multistratifiledkfold的な使う
    * sys.path.append('../input/iterative-stratification/iterative-stratification-master')
    *  kfold = MultilabelStratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=True)
    * [https://www.kaggle.com/gogo827jz/moa-public-pytorch-node](https://www.kaggle.com/gogo827jz/moa-public-pytorch-node)
- Out of Fold(oof)なファイルを必ず作成
- みんながsubmitしやすいようなnotebookを作成

## ToDo
- [https://www.kaggle.com/c/lish-moa/discussion/184005](https://www.kaggle.com/c/lish-moa/discussion/184005)
* データの前処理
- [ ] label smooth
- [x] ~~ トレーニングデータからコントロールグループを削除するとCV上がるらしい.LBは下がるけどブレンドするとLBもup ~~
- [ ] pretictのlowerとupperをclip
- [x] ~~ コントロールの出力を全て確認。もしかしたらすべて0かも ~~
- [ ] Pseudolabeling/noisy label training
- [ ] バランシングの適応 (優先順位低め)
- [ ] アップサンプリング[https://www.kaggle.com/c/lish-moa/discussion/187419](https://www.kaggle.com/c/lish-moa/discussion/187419)
    * [ノートブック](https://www.kaggle.com/tolgadincer/upsampling-multilabel-data-with-mlsmote)
    * [CVの方法のノートブック](https://www.kaggle.com/tolgadincer/mlsmote) (優先順位低め)
- [ ] ImbalancedDataSampler[pytorchの実装github](https://github.com/ufoym/imbalanced-dataset-sampler)
* 特徴量作成
* 成田タスク
    - [ ] ノンスコアのターゲットを予測し、その後のモデルのメタ特徴として使用
    - [ ] ノンスコアのターゲットも含めたモデルで学習する
* 畔栁タスク
    - [x] ~~ TabNetベースライン作成 ~~
    - [ ] カテゴリ変数を埋め込み特徴量として学習
    - [ ] noisy label training
    - [ ] AEでデノイズor中間層を特徴量に追加(Nakayamaさんこれ好きなイメージ)
    - [ ] メトリックラーニングをAEに適応して、他のモデルの特徴量にする (クラスタリングの重心を特徴量に加えるイメージ)
    * モデル構造
    - [ ] Tabunetの中間層でグループを作成すると,全結合のみよりもCVがgood
    - [ ] split NN keras[https://www.kaggle.com/gogo827jz/split-neural-network-approach-tf-keras](https://www.kaggle.com/gogo827jz/split-neural-network-approach-tf-keras)
    - [ ] TabNet num_decision_steps = 1 makes OOF score much better [https://www.kaggle.com/gogo827jz/moa-stacked-tabnet-baseline-tensorflow-2-0](https://www.kaggle.com/gogo827jz/moa-stacked-tabnet-baseline-tensorflow-2-0)
* 西山タスク
    - [ ] SVM
    - [ ] lgbm
    - [ ] xgb
    - [ ] kernel Ridge [https://www.kaggle.com/gogo827jz/kernel-logistic-regression-one-for-206-targets](https://www.kaggle.com/gogo827jz/kernel-logistic-regression-one-for-206-targets)
    - [ ] label smooth 
    - [ ] 計算量軽めモデルのハイパラチューニング[すごい重要らしい](https://www.kaggle.com/c/lish-moa/discussion/180918#1000976)
    - [ ] label powerset (マルチラベルタスクをマルチクラスタスクに変換するらしい. これを使えばSVMやlgbも一つのモデルでOKなのでは?)[http://scikit.ml/api/skmultilearn.problem_transform.lp.html](http://scikit.ml/api/skmultilearn.problem_transform.lp.html)
    - [ ] トレーニングデータからコントロールグループを削除するとCV上がるらしい.LBは下がるけどブレンドするとLBもup

## On Going
- TabNetベースライン作成
## Done
- Neural Oblivious Decision Ensembles(NNで作った多段決定木をredualに結合して出力)


## introduction
ホームディレクトリに.kaggleディレクトリが作成されている前提で作成します。
ない場合は、こちら[https://www.currypurin.com/entry/2018/kaggle-api](https://www.currypurin.com/entry/2018/kaggle-api)を参照してください。
```
# リポジトリのクローン
git clone https://github.com/ibkuroyagi/Mechanisms-of-Action-Prediction.git
# 仮想環境の構築
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# 入力データのダウンロード
mkdir input
cd input
kaggle competitions download -c lish-moa
unzip lish-moa
kaggle datasets download -d yasufuminakama/iterative-stratification
unzip iterative-stratification.zip
rm lish-moa.zip iterative-stratification.zipsssssss
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
    * Adversarial Validarionそんなにいらない
    * PCAは効かない
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
</div></details>

### External code
For those using Pytorch, here is an interesting thread about label smoothing: [https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/166833](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/166833)
And an implementation found [https://github.com/pytorch/pytorch/issues/7455](https://github.com/pytorch/pytorch/issues/7455)
```
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
```

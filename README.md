# Mechanisms of Action (MoA) Prediction
## Final submission deadline.
- November 30, 2020

## Data
In this competition, you will be predicting multiple targets of the Mechanism of Action (MoA) response(s) of different samples (sig_id), given various inputs such as gene expression data and cell viability data.

Two notes:

the training data has an additional (optional) set of MoA labels that are not included in the test data and not used for scoring.
the re-run dataset has approximately 4x the number of examples seen in the Public test.
Files
train_features.csv - Features for the training set. Features g- signify gene expression data, and c- signify cell viability data. cp_type indicates samples treated with a compound (cp_vehicle) or with a control perturbation (ctrl_vehicle); control perturbations have no MoAs; cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low).
train_targets_scored.csv - The binary MoA targets that are scored.
train_targets_nonscored.csv - Additional (optional) binary MoA responses for the training data. These are not predicted nor scored.
test_features.csv - Features for the test data. You must predict the probability of each scored MoA for each row in the test data.
sample_submission.csv - A submission file in the correct format.
### 日本語訳
本コンテストでは、遺伝子発現データや細胞生存率データなどの様々なインプットを与えられた異なるサンプル（sig_id）の作用機序（MoA）応答の複数のターゲットを予測

2つの注意点がある

トレーニングデータには、テストデータには含まれず、スコアリングには使用されないMoAラベルの追加（オプション）セットがある
再実行データセットは、パブリックテストで見られる例の約4倍の数を持つ
### ファイル
- train_features.csv - 訓練セットの特徴量．cp_type は化合物（cp_vehicle）または対照摂動（ctrl_vehicle）で処理されたサンプルを示し、対照摂動は MoA を持たない
- train_targets_scored.csv - スコアされるバイナリMoAターゲット
- train_targets_nonscored.csv - 訓練データの追加の（オプションの）バイナリMoA反応。これらは予測もスコア化もされない
- test_features.csv - テストデータの特徴量．テストデータの各行のスコアされたMoAの確率を予測する必要がある
- sample_submission.csv - 正しい形式の提出ファイル

## 評価指標
各カラムにおけるバイナリクロスエントロピーを計算しその平均値を、すべてのサンプルで平均した値

## ToDo
- 

# Soccer Prediction Research — OOS評価仕様書

## 1. 基本方針

評価は **chronological walk-forward OOS** を正本とする。

未来データを使ってモデル・特徴量・calibration・重みを選択し、その同じOOSで成績を報告することは禁止する。

## 2. Walk-forward

基本形：

```text
Train: 過去データ
Validation: train期間内の未来側
Test/OOS: その時点で完全に未知の期間

評価後
↓
train windowを時間方向に拡張
↓
次のOOS
↓
繰り返し
```

すべてのsplitは時系列順。random train/test splitは禁止。

同一日の試合もkickoff UTC順で扱う。

## 3. OOSの原則

各OOS rowでは、その時点以前に利用可能だったデータだけで以下を行う。

1. feature generation
2. model fitting
3. hyperparameter/model selection
4. calibration fitting
5. ensemble weighting
6. prediction
7. actual result取得

OOS actualをmodel selectionに先に使用してはならない。

## 4. 1X2評価

### Primary
- Multiclass LogLoss

### Secondary
- Accuracy
- Multiclass Brier Score
- Home/Draw/Away別LogLoss
- calibration / reliability
- ECE等のcalibration error

表示する確率はHome / Draw / Awayの3値とし、原則として合計1となる正規化probabilityを保存する。

## 5. Score評価

最低限以下を評価する。

- Home goals MAE
- Away goals MAE
- Total goals MAE
- Exact Score Hit Rate
- Top-3 Score Hit Rate
- Top-4 Score Hit Rate

ユーザー向けscore predictionは3〜4候補を表示する。候補確率は「候補集合の確率を再正規化するか」「元のjoint probabilityを表示するか」を実装前に統一する。現時点では未確定として、バックテスト実装時に仕様化する。

## 6. MOM評価

MOM targetを取得できるcompetition/sourceだけ評価する。

- Top-1 Hit Rate
- Top-3 Hit Rate

MOMの正解データまたはavailability timestampが不足するcompetitionでは評価・予測を無理に生成しない。

## 7. Low / High

Low/Highを実装する場合、まずtarget定義を固定する。

例としてtotal goalsに対するthreshold市場なら、targetは実際のtotal goalsから直接生成する。

定義が固定される前にモデルを作り、結果に合わせてthresholdを変更することは禁止する。

## 8. Calibration評価

candidate calibrationはchronological validationで比較する。

候補例：

- uncalibrated
- temperature scaling
- Platt scaling（対象形式に適用可能な場合）
- isotonic regression

calibration parameterは過去データだけでfitする。

最終OOSはcalibrator選択に使わない。

## 9. モデル候補

初期baselineとcandidateを同じOOS splitで比較する。

モデル種類は実データcoverageを確認してから決定する。単一modelへの固定を避け、logistic/linear系、tree ensemble、gradient boosting、Poisson系等を候補として検証できる構造にする。

ただし「モデル数が多いこと」を精度とみなさない。OOSで改善しないモデルは採用しない。

## 10. 改善の採用ルール

研究サイクルは以下を固定する。

```text
Current baseline
    ↓
弱点を特定
    ↓
改善candidateを作る
    ↓
同一chronological OOSで比較
    ↓
metrics / segment / calibration / robustnessを確認
    ↓
改善が明確 → adopt
改善なし・悪化 → reject
不明確 → hold / 追加検証
```

特に **OOSで確認できない改善は採用しない**。

## 11. Primary metricの採用ルール

1X2ではLogLossをprimaryとし、Accuracyだけを理由に採用しない。

採用判断では少なくとも以下を同時確認する。

- primary metric
- Brier
- calibration
- Accuracy
- competition/league segment
- 時系列期間別安定性

score/MOM/Low-Highについても各市場のprimary metricを事前に固定する。

## 12. Segment評価

全体平均だけでなく、最低限以下を分離する。

- competition
- season
- league tier（coverageが十分な場合）
- home/away strength bucket等、実装時に事前定義したsegment

segmentごとのサンプル数も必ず保存する。

## 13. 統計的安定性

1回のOOS差だけで採用しない。複数OOS periodで同方向の改善があるか確認する。

confidence interval / bootstrap等を利用できる場合は、差分の不確実性も保存する。

採用の最低改善量・CI thresholdはデータcoverageとsample sizeを監査してから確定する。データを見る前に都合のよい閾値を設定しない。

## 14. OOS結果保存

各predictionについて最低限保存する。

- match_id
- kickoff_utc
- prediction_cutoff_at_utc
- model_version
- data_snapshot_id
- p_home / p_draw / p_away
- score candidates
- MOM candidates
- actual result
- actual score
- metric inputs
- leakage_gate_status

集計結果にはrun ID、git commit SHA、dataset snapshot/hash、model configurationを記録する。

## 15. 再現性

同じ

`code commit + dataset snapshot + configuration + OOS split definition`

から同じ結果を再現できることを目標とする。

外部sourceが更新される場合、最新データで過去結果を上書きせずsnapshot/hashを保持する。

## 16. 研究ループ

```text
最新データ取得
↓
データ品質検査
↓
point-in-time / leakage gate
↓
feature generation
↓
chronological walk-forward OOS
↓
1X2 / Score / MOM / Low-High評価
↓
calibration評価
↓
weakness discovery
↓
candidate improvement
↓
同一OOSで再評価
↓
改善が明確なら採用
↓
model/version/snapshot保存
↓
次の研究サイクル
```

このループは本番予測と研究評価を混同しないようにする。

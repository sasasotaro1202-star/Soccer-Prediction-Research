# Soccer Prediction Research — 時刻・リーク仕様書

## 1. 最重要原則

**feature_available_at <= prediction_cutoff_at** を全featureに対して満たさなければ、そのfeatureを予測入力として使用しない。

「試合日より前に公開されたはず」「現在のDBに入っているから使える」といった推測は禁止する。

## 2. 4つの時刻

各データについて最低限次を分離する。

- `event_time_utc`: 実際の試合・イベントが発生した時刻
- `source_available_at_utc`: その情報が外部sourceで利用可能になった時刻
- `retrieved_at_utc`: システムが取得した時刻
- `prediction_cutoff_at_utc`: その予測で利用を許可する締切時刻

`retrieved_at`は「sourceで公開された時刻」の代用にはならない。

## 3. Prediction cutoff

運用と研究を分離する。

### A. Operational cutoff

再現可能な固定cutoffを使用する。候補として`kickoff - 60 minutes`を基本案とするが、最終値はデータcoverage監査後に確定する。

### B. Latest-available cutoff

実運用のprediction timestampをcutoffとし、その時点までに取得・保存されたsnapshotだけを使用する。

過去データにsource availability timestampがない場合、過去のlatest-available状態を正確に再現できないため、historical backtestではその制約を明示する。

## 4. Point-in-time feature rule

feature生成は次の条件を満たすレコードだけから行う。

```text
source_available_at_utc <= prediction_cutoff_at_utc
AND event_time_utc < prediction_cutoff_at_utc
```

例外はfixtureそのものなど、未来の予定情報である。この場合も「fixtureがcutoff前に公開されていた」ことを別途確認する。

## 5. 絶対に禁止するリーク

### 結果リーク
- future match result
- future score
- future xG
- future cards/corners/shots
- match終了後に更新されたteam/player statistics

### 集計リーク
- season終了後のstandings
- 現在時点で再計算されたrating
- 全seasonを使った平均値
- future matchesを含むrolling average

### 市場リーク
- closing oddsを早いprediction cutoffへ遡って投入
- kickoff後に取得したodds
- prediction後に市場から更新された情報

### lineup/playerリーク
- lineup発表前の確定lineupをfeature扱いすること
- injury情報の公開時刻を確認せず使用すること
- match後に確定した出場時間・performanceをmatch前featureに使用すること

## 6. Rolling / expanding feature

team form、Elo、xG/xGA、shots、goals、cards等は各matchのprediction cutoff時点で再計算する。

例えば直近5試合formは、対象matchより前に完了した最大5試合だけを使用する。

```text
match t
  ↓
state(t) = information available strictly before cutoff(t)
  ↓
predict(t)
  ↓
actual(t) is revealed
  ↓
state(t+1) にのみ追加
```

同一matchのactualを、そのmatch自身のfeature計算に戻してはならない。

## 7. 同日試合

同一日に複数試合がある場合、日付だけでtrain/testを分割しない。kickoff UTC順で処理する。

先に開始した試合の結果を、後に開始する試合のprediction cutoffより前に実際に利用可能だった場合だけ後続試合に利用可能とする。ただしsource_available_atを確認できない場合は保守的に除外する。

## 8. Standings / Elo / team state

過去時点のstateをexpanding updateで再構築する。現在のstandingsや現在のEloを過去にコピーする実装は禁止。

## 9. Calibration leakage

Calibration modelも本体modelと同様にchronologicalに扱う。

- calibration parameterをfuture OOSから決定しない
- OOS評価対象をcalibrator選択に再利用しない
- candidate calibrationの選択用validationと最終OOSを分離する
- temperature scaling / isotonic / Platt等を採用する場合も過去データだけでfitする

## 10. Leakage gate

各prediction rowについて次を保存する。

- `feature_cutoff_max_utc`
- `prediction_cutoff_at_utc`
- `leakage_gate_status`
- `leakage_gate_reason`
- `data_snapshot_id`

次を満たさないrunはfail。

```text
feature_cutoff_max_utc <= prediction_cutoff_at_utc
```

さらにfuture-derived columns、重複match、異常timestampを検査する。

## 11. Champions League target rule

ユーザー指定の結果定義を固定する。

- regulationでHome/Away勝利 → その勝者
- extra timeで決着 → その勝者をHome/Away win
- penalty shootoutで決着 → Draw

raw dataではregulation/extra-time/penaltyの情報を可能な限り分離保存し、target transformationを明示的に記録する。

## 12. 予測保存

各predictionは最低限以下を保存する。

`match_id, competition, season, kickoff_utc, kickoff_jst, prediction_cutoff_at_utc, model_version, data_snapshot_id, feature_cutoff_max_utc, leakage_gate_status`

加えて1X2確率、score候補、MOM候補、calibration versionを保存する。

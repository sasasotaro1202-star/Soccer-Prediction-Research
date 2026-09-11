# Soccer Prediction Research — データ取得仕様書

## 1. 目的

本仕様書は、試合予測・バックテスト・自動研究で使用するデータについて、取得元、用途、時刻情報、保存単位、採用条件を固定するための仕様である。

原則は **「取得できるデータ」ではなく「予測時点で利用可能だったことを証明できるデータ」だけを予測特徴量に採用する**。

## 2. データソース方針

| 区分 | 第一候補 | 用途 | 本番採用条件 |
|---|---|---|---|
| 試合日程・結果 | football-data.org v4 | fixture、kickoff、status、competition、season、team、result | source時刻を記録し、結果は試合終了後データとしてのみ使用 |
| 歴史結果・試合統計 | Football-Data.co.uk | 過去結果、shots、corners、cards等、バックテスト用データ | 対象リーグ・seasonのcoverageと欠損を検査 |
| 歴史オッズ | Football-Data.co.uk | market benchmark、条件付き市場特徴量 | オッズ取得時刻をprediction cutoffと比較できる場合のみ特徴量化 |
| event / shot / lineup | StatsBomb Open Data | event、shot、lineup、360が提供される対象大会・season | coverage確認、利用規約順守、point-in-time利用可否を別監査 |
| xG | 別途検証する公開ソース | xG/xGA等 | source_available_atを証明できるまで本番特徴量にしない |
| Understat | 別途監査 | xG等の補助データ候補 | 「Understat=0」の原因を調査し、取得・mapping・時刻・欠損を確認後に判断 |

football-data.org v4はcompetition/match/team/standing/person等を提供し、matchには`utcDate`とstatus等が含まれる。無料登録プランは10 requests/minuteと文書化されているため、取得処理はcacheを前提とする。citeturn0search1turn0search7

Football-Data.co.ukは無料の歴史結果・match statistics・oddsを提供し、2019/20以降はmarket opening後に取得したoddsとclosing oddsの2系列がある。ただしclosing oddsは予測時点より後に確定する可能性があるため、pre-match featureとして無条件に使用してはならない。citeturn0search0turn0search3

Football-Data.co.ukのfixture/oddsページでは、週末fixtureは通常金曜17:00 BSTまで、midweek fixtureは火曜13:00 BSTまでに収集されると説明されている。この時刻は「そのデータがその時点で存在した」ことを判断する際のsource-specific metadataとして保存する。citeturn0search10

StatsBomb Open Dataはcompetition/season、matches、events、lineups、選択された360 dataをJSONで提供する。利用時はStatsBombの出典表示条件を守る。citeturn0search4

## 3. 保存する基本ID

すべての取得データは可能な限り以下を保持する。

- `source_name`
- `source_record_id`
- `competition_id`
- `season_id`
- `match_id`
- `home_team_id`
- `away_team_id`
- `event_time_utc`
- `source_available_at_utc`
- `retrieved_at_utc`
- `prediction_cutoff_at_utc`
- `source_url_or_endpoint`
- `source_version_or_hash`
- `raw_snapshot_id`

外部sourceのIDがない場合は、正規化したcompetition/season/kickoff/teamの組み合わせを仮キーとして保存し、重複・mapping監査を行う。

## 4. 時刻標準

- 内部時刻は **timezone-aware UTC** を正本とする。
- ユーザー向け表示はUTCからJSTへ変換する。
- naive datetimeは禁止。
- `kickoff_jst`は表示用derived fieldであり、モデルの正本時刻は`kickoff_utc`。
- 同一日の複数試合もkickoff UTCの完全な順序で処理する。

## 5. データ取得頻度

### 定期取得
- fixture/status: 少なくとも1日数回。運用時はprediction horizonに応じて増頻。
- historical result/statistics: 日次または新season/新データ公開時。
- odds: sourceの更新タイミングに合わせて取得し、snapshotを保存。
- event/lineup: 利用可能なopen-data更新後に取得。

### 取得時の原則

同じURLを再取得して上書きするのではなく、可能な限りsnapshotとして保存する。これにより、後から「予測時点で何が見えていたか」を再現できるようにする。

## 6. データ品質ゲート

予測・backtest開始前に以下を検査する。

1. match ID重複
2. team ID/name mapping不整合
3. kickoff timezone異常
4. scoreの不正値
5. competition/season不整合
6. 欠損率
7. 同一試合への複数snapshotの競合
8. `source_available_at > prediction_cutoff_at`
9. 未来試合の結果がfeatureに混入していないか
10. probabilityの非有限値・範囲外

致命的エラーはrunをfailさせ、silent repairは禁止する。

## 7. 市場オッズの扱い

closing oddsは原則としてprediction featureから除外する。予測cutoff以前に実際に利用可能だったopening/early oddsについてのみ、availability timestampが検証できる場合にfeature化する。

時刻が検証できない場合は、oddsを「予測モデル入力」ではなく「事後的なmarket benchmark」として利用する。

## 8. Current standings / ratingの扱い

現在の順位表・ratingを過去試合へ遡って直接投入してはならない。backtestでは各prediction cutoff時点までの試合からrolling/expanding stateを再構築する。

## 9. 未確定項目

以下はデータcoverage監査後に確定する。

- xGの本番採用source
- Understat取得方式と0件の原因
- lineup/injuryのpoint-in-time source
- 固定prediction cutoffの最終運用値
- 各competitionの最低coverage threshold

未確定項目を推測で埋めて実装しない。

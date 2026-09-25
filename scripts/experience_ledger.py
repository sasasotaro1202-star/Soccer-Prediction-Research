"""Durable prediction/outcome experience ledger.

Records distinct PIT-safe prediction states, settles completed matches from free public
sources, and generates rolling/segmented performance metrics. Unknown outcomes stay
unsettled; missing labels are never converted to negatives.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.data.external_fetch import ExternalFetcher
from src.data.sofascore_mom_labels import fetch_mom_label

LEDGER = Path("data/experience/prediction_ledger.csv")
METRICS = Path("artifacts/experience_metrics.csv")
STATUS = Path("artifacts/experience_status.json")

def _now():
    return pd.Timestamp(datetime.now(timezone.utc))

def _norm(value: Any) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", str(value or "").casefold()).strip()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] not in {"P", "Z"})

def _key(kickoff: Any, home: Any, away: Any) -> str:
    ts = pd.to_datetime(kickoff, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return f"{pd.Timestamp(ts).isoformat()}|{_norm(home)}|{_norm(away)}"

def _hash_state(row: pd.Series) -> str:
    cols = ["match_id","kickoff_utc","home_team","away_team","competition","model_version",
            "p_home","p_draw","p_away","score_1","score_1_probability","score_2","score_2_probability",
            "score_3","score_3_probability","mom_1_player_id","mom_1_probability","mom_2_player_id",
            "mom_2_probability","mom_3_player_id","mom_3_probability","mom_4_player_id","mom_4_probability"]
    payload = {}
    for c in cols:
        v = row.get(c, "")
        if pd.isna(v):
            v = ""
        payload[c] = str(round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]

def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() and path.stat().st_size else pd.DataFrame()

def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)

def record_prediction_file(predictions_path: str, prediction_time: str | None = None) -> dict:
    src = Path(predictions_path)
    if not src.is_file() or src.stat().st_size == 0:
        return {"status":"NO_PREDICTIONS","added":0,"ledger_rows":len(_read(LEDGER))}
    incoming = pd.read_csv(src)
    if incoming.empty:
        return {"status":"NO_PREDICTIONS","added":0,"ledger_rows":len(_read(LEDGER))}
    required = {"match_id","kickoff_utc","home_team","away_team","competition","p_home","p_draw","p_away","model_version"}
    missing = sorted(required - set(incoming.columns))
    if missing:
        raise RuntimeError(f"prediction ledger input missing required columns: {missing}")
    incoming = incoming.copy()
    incoming["prediction_recorded_at_utc"] = pd.Timestamp(prediction_time, tz="UTC") if prediction_time else _now()
    for c in ["p_home","p_draw","p_away","score_1_probability","score_2_probability","score_3_probability",
              "mom_1_probability","mom_2_probability","mom_3_probability","mom_4_probability"]:
        if c in incoming:
            incoming[c] = pd.to_numeric(incoming[c], errors="coerce").round(5)
    incoming["fixture_key"] = incoming.apply(lambda r: _key(r["kickoff_utc"],r["home_team"],r["away_team"]), axis=1)
    incoming["prediction_state_id"] = incoming.apply(_hash_state, axis=1)
    existing = _read(LEDGER)
    seen = set(existing["prediction_state_id"].astype(str)) if not existing.empty and "prediction_state_id" in existing else set()
    new_rows = incoming[~incoming["prediction_state_id"].astype(str).isin(seen)].copy()
    for c in ["actual_home_goals","actual_away_goals","actual_result","actual_score","settled_at_utc","settlement_source",
              "correct_1x2","predicted_1x2","confidence","score_top1_hit","score_top3_hit",
              "over_2_5_correct","btts_correct","mom_actual_player_id","mom_top1_hit","mom_top4_hit","mom_settlement_status"]:
        if c not in new_rows:
            new_rows[c] = pd.NA
    combined = pd.concat([existing,new_rows],ignore_index=True) if not existing.empty else new_rows
    if not combined.empty:
        combined = combined.drop_duplicates("prediction_state_id",keep="first").sort_values(
            ["kickoff_utc","prediction_recorded_at_utc","match_id"],kind="mergesort")
    _write_csv(LEDGER, combined)
    return {"status":"RECORDED","added":len(new_rows),"ledger_rows":len(combined),
            "unique_fixtures":int(combined["match_id"].nunique()) if not combined.empty else 0}

def _score_value(score_obj):
    if not isinstance(score_obj,dict): return None
    for k in ("normaltime","current","period2"):
        try:
            if score_obj.get(k) is not None: return int(score_obj[k])
        except (TypeError,ValueError): pass
    return None

def _event_outcome(event):
    st=str((event.get("status") or {}).get("type") or "").lower()
    if st not in {"finished","afterextra","afterextratime","afterpenalties"}: return None,None,st
    return _score_value(event.get("homeScore")),_score_value(event.get("awayScore")),st

def _event_key(event):
    try: kickoff=pd.to_datetime(int(event["startTimestamp"]),unit="s",utc=True)
    except (KeyError,TypeError,ValueError,OverflowError): return ""
    return _key(kickoff,(event.get("homeTeam") or {}).get("name",""),(event.get("awayTeam") or {}).get("name",""))

def _sofa_match_id(event):
    tournament=event.get("tournament") or {}
    ut=tournament.get("uniqueTournament") or event.get("uniqueTournament") or tournament
    comp=str(ut.get("slug") or ut.get("name") or "")
    canonical=f"sofascore|{event.get('id')}|{comp}|{(event.get('homeTeam') or {}).get('name','')}|{(event.get('awayTeam') or {}).get('name','')}"
    return "sofa:"+hashlib.sha256(canonical.encode()).hexdigest()[:20]

def _espn_match_id(event): return f"espn:{event.get('id')}"

def _fetch_sofa_events(dates, fetcher):
    events=[]; latest=""; seen=set()
    for date in dates:
        response=fetcher.get("sofascore_scheduled_events",
            f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{date}",
            headers={"User-Agent":"Soccer-Prediction-Research/experience-ledger"},cache_ttl_seconds=900)
        latest=max(latest,response.metadata.retrieved_at)
        payload=json.loads(response.body.decode("utf-8"))
        for e in payload.get("events",[]) or []:
            if not isinstance(e,dict): continue
            eid=str(e.get("id") or "")
            if not eid or eid in seen: continue
            seen.add(eid); events.append(e)
    return events,latest

def _fetch_espn_events(dates, fetcher):
    leagues={"EPL":"eng.1","ERE":"ned.1","LL":"esp.1","SA":"ita.1","BL1":"ger.1","FL1":"fra.1",
             "J1":"jpn.1","J2":"jpn.2","J3":"jpn.3","UCL":"uefa.champions","UEL":"uefa.europa","MLS":"usa.1"}
    events=[]
    for date in dates:
        for league in leagues.values():
            try:
                r=fetcher.get("espn_scoreboard",
                    f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard",
                    params={"dates":date.replace("-","")},
                    headers={"User-Agent":"Soccer-Prediction-Research/experience-ledger"},cache_ttl_seconds=900)
                p=json.loads(r.body.decode("utf-8")); events.extend([e for e in p.get("events",[]) or [] if isinstance(e,dict)])
            except Exception: continue
    return events

def _espn_event_key(e):
    comps=e.get("competitions") or []
    if not comps: return ""
    comp=comps[0]
    hs=next((x for x in comp.get("competitors",[]) or [] if x.get("homeAway")=="home"),{})
    aw=next((x for x in comp.get("competitors",[]) or [] if x.get("homeAway")=="away"),{})
    return _key(e.get("date") or comp.get("startDate"),
                (hs.get("team") or {}).get("displayName") or hs.get("id"),
                (aw.get("team") or {}).get("displayName") or aw.get("id"))

def _settle_row(row, by_key, by_id):
    event=by_id.get(str(row.get("match_id") or "")) or by_key.get(str(row.get("fixture_key") or ""))
    if event is None: return {}
    if str(event.get("id","")).isdigit():
        pass
    if "competitions" in event:
        comp=(event.get("competitions") or [{}])[0]
        h=next((x for x in comp.get("competitors",[]) or [] if x.get("homeAway")=="home"),{})
        a=next((x for x in comp.get("competitors",[]) or [] if x.get("homeAway")=="away"),{})
        hs=h.get("score"); aws=a.get("score")
        try: hg,ag=int(hs),int(aws)
        except (TypeError,ValueError): hg,ag=None,None
        status=(comp.get("status") or event.get("status") or {})
        status_type=str((status.get("type") or {}).get("name") or status.get("type") or "").lower()
    else:
        hg,ag,status_type=_event_outcome(event)
    if hg is None or ag is None: return {}
    actual="H" if hg>ag else "A" if hg<ag else "D"
    p=np.array([float(row["p_home"]),float(row["p_draw"]),float(row["p_away"])])
    pred=["H","D","A"][int(p.argmax())]
    top=[str(row.get(f"score_{i}","")).strip() for i in (1,2,3)]
    out={"actual_home_goals":hg,"actual_away_goals":ag,"actual_result":actual,"actual_score":f"{hg}-{ag}",
         "settled_at_utc":_now().isoformat(),"settlement_source":"espn" if str(row["match_id"]).startswith("espn:") else "sofascore",
         "correct_1x2":int(pred==actual),"predicted_1x2":pred,"confidence":float(p.max()),
         "score_top1_hit":int(top[0]==f"{hg}-{ag}"),"score_top3_hit":int(f"{hg}-{ag}" in top)}
    if pd.notna(row.get("market_over_2_5")): out["over_2_5_correct"]=int((float(row["market_over_2_5"])>=0.5)==((hg+ag)>2.5))
    if pd.notna(row.get("market_btts_yes")): out["btts_correct"]=int((float(row["market_btts_yes"])>=0.5)==(hg>0 and ag>0))
    return out

def settle_predictions(days_back=14):
    ledger=_read(LEDGER)
    if ledger.empty: return {"status":"NO_LEDGER","settled":0,"pending":0}
    now=_now(); cutoff=now-pd.Timedelta(days=max(1,int(days_back)))
    ledger["kickoff_utc"]=pd.to_datetime(ledger["kickoff_utc"],utc=True,errors="coerce")
    if "actual_result" not in ledger: ledger["actual_result"]=pd.NA
    pending=ledger[ledger["kickoff_utc"].notna() & (ledger["kickoff_utc"]<=now) &
                   (ledger["kickoff_utc"]>=cutoff) & ledger["actual_result"].isna()].copy()
    if pending.empty:
        compute_metrics(ledger); return {"status":"NOTHING_TO_SETTLE","settled":0,"pending":0}
    dates=sorted(set(pending["kickoff_utc"].dt.strftime("%Y-%m-%d").tolist()))
    fetcher=ExternalFetcher(cache_dir="cache/external",timeout=20,retries=3,backoff=1.0)
    sofa_events,retrieval=_fetch_sofa_events(dates,fetcher)
    espn_events=_fetch_espn_events(dates,fetcher)
    by_key={}; by_id={}
    for e in sofa_events:
        k=_event_key(e)
        if k: by_key[k]=e
        by_id[_sofa_match_id(e)]=e
    for e in espn_events:
        k=_espn_event_key(e)
        if k: by_key[k]=e
        by_id[_espn_match_id(e)]=e
    settled=0
    for idx in pending.index:
        changes=_settle_row(ledger.loc[idx],by_key,by_id)
        if not changes: continue
        for k,v in changes.items(): ledger.at[idx,k]=v
        event=by_id.get(str(ledger.at[idx,"match_id"])) or by_key.get(str(ledger.at[idx,"fixture_key"]))
        if event is not None and any(pd.notna(ledger.at[idx,f"mom_{r}_player_id"]) for r in range(1,5)):
            try:
                eid=event.get("id")
                if eid:
                    label=fetch_mom_label(str(eid),fetcher=fetcher)
                    if label.get("label_found"):
                        actual_player=str(label["player_id"])
                        ledger.at[idx,"mom_actual_player_id"]=actual_player
                        ledger.at[idx,"mom_top1_hit"]=int(str(ledger.at[idx,"mom_1_player_id"])==actual_player)
                        ledger.at[idx,"mom_top4_hit"]=int(actual_player in {str(ledger.at[idx,f"mom_{r}_player_id"]) for r in range(1,5)})
                        ledger.at[idx,"mom_settlement_status"]="OFFICIAL"
            except Exception:
                ledger.at[idx,"mom_settlement_status"]="UNAVAILABLE"
        settled+=1
    _write_csv(LEDGER,ledger)
    metrics=compute_metrics(ledger)
    return {"status":"SETTLED","settled":settled,"pending_after":int(ledger["actual_result"].isna().sum()),
            "retrieval_at_utc":retrieval,"metrics_rows":metrics}

def compute_metrics(ledger=None):
    if ledger is None: ledger=_read(LEDGER)
    METRICS.parent.mkdir(parents=True,exist_ok=True)
    if ledger.empty:
        pd.DataFrame([{"scope":"all","n":0,"status":"NO_SETTLED_PREDICTIONS"}]).to_csv(METRICS,index=False); return 1
    d=ledger.copy(); d["kickoff_utc"]=pd.to_datetime(d["kickoff_utc"],utc=True,errors="coerce")
    d=d[d["actual_result"].notna()].copy()
    if d.empty:
        pd.DataFrame([{"scope":"all","n":0,"status":"NO_SETTLED_PREDICTIONS"}]).to_csv(METRICS,index=False); return 1
    rows=[]
    def emit(scope,x,label=""):
        if x.empty:return
        r={"scope":scope,"segment":label,"n":len(x),
           "1x2_accuracy_pct":round(pd.to_numeric(x["correct_1x2"],errors="coerce").mean()*100,4),
           "score_top1_accuracy_pct":round(pd.to_numeric(x["score_top1_hit"],errors="coerce").mean()*100,4),
           "score_top3_accuracy_pct":round(pd.to_numeric(x["score_top3_hit"],errors="coerce").mean()*100,4)}
        for col,name in [("over_2_5_correct","over_2_5_accuracy_pct"),("btts_correct","btts_accuracy_pct"),
                         ("mom_top1_hit","mom_top1_accuracy_pct"),("mom_top4_hit","mom_top4_accuracy_pct")]:
            vals=pd.to_numeric(x[col],errors="coerce").dropna() if col in x else pd.Series(dtype=float)
            if not vals.empty:r[name]=round(vals.mean()*100,4)
        for cls,name in [("H","home"),("D","draw"),("A","away")]:
            vals=pd.to_numeric(x.loc[x["actual_result"]==cls,"correct_1x2"],errors="coerce").dropna()
            if not vals.empty:r[name+"_hit_rate_pct"]=round(vals.mean()*100,4)
        r["models"]="|".join(sorted(set(x["model_version"].dropna().astype(str)))) if "model_version" in x else ""
        r["competitions"]=int(x["competition"].nunique()) if "competition" in x else 0
        rows.append(r)
    now=_now()
    for scope,start in [("all",d["kickoff_utc"].min()),("365d",now-pd.Timedelta(days=365)),("90d",now-pd.Timedelta(days=90)),
                        ("30d",now-pd.Timedelta(days=30)),("7d",now-pd.Timedelta(days=7))]:
        emit(scope,d[d["kickoff_utc"]>=start])
    for model,x in d.groupby("model_version",dropna=False): emit("model",x,str(model))
    for comp,x in d.groupby("competition",dropna=False): emit("competition",x,str(comp))
    _write_csv(METRICS,pd.DataFrame(rows))
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps({"status":"OK","settled_predictions":len(d),"ledger_rows":len(ledger),
                                  "generated_at_utc":_now().isoformat(),"metrics_file":str(METRICS)},indent=2),encoding="utf-8")
    return len(rows)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("record"); a.add_argument("--predictions",default="artifacts/predictions.csv"); a.add_argument("--prediction-time")
    b=sub.add_parser("settle"); b.add_argument("--days-back",type=int,default=14)
    sub.add_parser("metrics"); args=p.parse_args()
    if args.command=="record": print(json.dumps(record_prediction_file(args.predictions,args.prediction_time),ensure_ascii=False,default=str))
    elif args.command=="settle": print(json.dumps(settle_predictions(args.days_back),ensure_ascii=False,default=str))
    else: print(json.dumps({"status":"METRICS","rows":compute_metrics()},ensure_ascii=False,default=str))
if __name__=="__main__": raise SystemExit(main())

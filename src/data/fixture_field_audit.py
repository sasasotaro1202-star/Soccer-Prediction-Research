from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
from src.data.competition_sources import source_plans
from src.data.football_data import load_available_history
TARGET_COMPETITIONS=("EPL","CHA","BL1","SA","LL","FL1","UCL","UEL","J1","J2","J3","DFBP","CAR","FRI")
COMPETITION_NAMES={"EPL":"Premier League","CHA":"Championship","BL1":"Bundesliga","SA":"Serie A","LL":"La Liga","FL1":"Ligue 1","UCL":"UEFA Champions League","UEL":"UEFA Europa League","J1":"J1","J2":"J2","J3":"J3","DFBP":"DFB-Pokal","CAR":"Carabao Cup / EFL Cup","FRI":"Club Friendlies"}
CANONICAL_FIELDS=("fixture_id","competition","season","home_team","away_team","kickoff_utc","result","home_goals","away_goals","home_shots","away_shots","home_shots_on_target","away_shots_on_target","home_corners","away_corners","home_fouls","away_fouls","home_yellow_cards","away_yellow_cards","home_red_cards","away_red_cards")
@dataclass(frozen=True)
class AuditConfig:
    start_year:int=2010; end_year:int=2025; prediction_cutoff_minutes:int=60

def _status_for_value(value,field):
    if pd.isna(value): return "MISSING"
    if field.startswith(("home_","away_")) and field.endswith(("goals","shots","shots_on_target","corners","fouls","yellow_cards","red_cards")):
        try:
            if float(value)==0: return "REAL_ZERO"
        except (TypeError,ValueError): pass
    return "AVAILABLE"

def _fixture_id(row): return f"{row.get('source_name','unknown')}:{row.get('source_record_id',row.get('match_id','unknown'))}"

def fixture_audit(history):
    cols=["fixture_id","competition","season","home_team","away_team","home_team_id","away_team_id","kickoff_utc","source","source_fixture_id","fixture_status","kickoff_precision","team_mapping_status"]
    if history.empty:return pd.DataFrame(columns=cols)
    rows=[]
    for _,r in history.iterrows():
        home=str(r.get("home_team","")).strip(); away=str(r.get("away_team","")).strip(); mapping="VALID" if home and away and home.lower()!=away.lower() else "INVALID"
        rows.append({"fixture_id":_fixture_id(r),"competition":r.get("competition"),"season":r.get("season"),"home_team":home,"away_team":away,"home_team_id":r.get("home_team_id",pd.NA),"away_team_id":r.get("away_team_id",pd.NA),"kickoff_utc":r.get("kickoff_utc"),"source":r.get("source_name"),"source_fixture_id":r.get("source_record_id",r.get("match_id")),"fixture_status":"OBSERVED","kickoff_precision":r.get("event_time_precision"),"team_mapping_status":mapping})
    return pd.DataFrame(rows)

def field_audit(history,config=AuditConfig()):
    rows=[]
    if history.empty:return pd.DataFrame()
    for _,r in history.iterrows():
        fid=_fixture_id(r); kickoff=pd.to_datetime(r.get("kickoff_utc"),utc=True,errors="coerce"); cutoff=kickoff-pd.Timedelta(minutes=config.prediction_cutoff_minutes) if pd.notna(kickoff) else pd.NaT; available=pd.to_datetime(r.get("source_available_at_utc"),utc=True,errors="coerce"); retrieved=pd.to_datetime(r.get("retrieved_at_utc"),utc=True,errors="coerce")
        for field in CANONICAL_FIELDS[6:]:
            value=r.get(field,pd.NA); value_status=_status_for_value(value,field)
            if pd.isna(cutoff): pit_status,pit_reason="PIT_UNKNOWN","Invalid kickoff/cutoff"
            elif pd.notna(available): pit_status,pit_reason=("PIT_SAFE","Explicit source availability timestamp") if available<=cutoff else ("PIT_UNSAFE","Source availability after cutoff")
            elif field in {"home_goals","away_goals","result"}: pit_status,pit_reason="PIT_UNSAFE","Own-match outcome is post-event information"
            else: pit_status,pit_reason="PIT_UNKNOWN","No source publication timestamp; retrieval time is not publication time"
            rows.append({"fixture_id":fid,"competition":r.get("competition"),"season":r.get("season"),"home_team":r.get("home_team"),"away_team":r.get("away_team"),"event_time_utc":kickoff,"prediction_cutoff_at_utc":cutoff,"field_name":field,"value":value,"value_status":value_status,"source":r.get("source_name"),"source_fixture_id":r.get("source_record_id",r.get("match_id")),"source_available_at_utc":available,"retrieved_at_utc":retrieved,"pit_status":pit_status,"pit_reason":pit_reason})
    return pd.DataFrame(rows)

def pit_audit(field_rows):
    cols=["competition","season","source","field_name","pit_status","count","rate"]
    if field_rows.empty:return pd.DataFrame(columns=cols)
    out=field_rows.groupby(["competition","season","source","field_name","pit_status"],dropna=False).size().reset_index(name="count"); totals=out.groupby(["competition","season","source","field_name"],as_index=False)["count"].sum().rename(columns={"count":"total"}); out=out.merge(totals,on=["competition","season","source","field_name"],how="left"); out["rate"]=out["count"]/out["total"]; return out

def coverage_matrix(history,field_rows,acquisition=None,config=AuditConfig()):
    rows=[]; observed=set(history["competition"].dropna().astype(str)) if not history.empty else set()
    for comp in TARGET_COMPETITIONS:
        ch=history[history["competition"].astype(str)==comp] if not history.empty else pd.DataFrame()
        if comp not in observed: rows.append({"competition":comp,"competition_name":COMPETITION_NAMES[comp],"season":pd.NA,"source":"current_observed_adapter","field":pd.NA,"status":"UNAVAILABLE","fixture_count":0,"reason":"No observed rows from current adapter; not a claim that no data exists elsewhere."})
        else: rows.append({"competition":comp,"competition_name":COMPETITION_NAMES[comp],"season":"ALL_OBSERVED","source":"current_observed_adapter","field":pd.NA,"status":"AVAILABLE","fixture_count":int(len(ch)),"reason":"Observed rows from current adapter"})
    if acquisition is not None and not acquisition.empty:
        for _,r in acquisition.iterrows():
            comp=str(r.get("competition")); raw=str(r.get("season"));
            if comp not in TARGET_COMPETITIONS: continue
            season=raw if "/" in raw or not raw.isdigit() else (raw if comp in {"J1","J2","J3"} else f"{raw}/{str(int(raw)+1)[-2:]}")
            rows.append({"competition":comp,"competition_name":COMPETITION_NAMES[comp],"season":season,"source":str(r.get("source","unknown")),"field":"__FIXTURE_CELL__","status":str(r.get("status","UNKNOWN")),"fixture_count":int(pd.to_numeric(r.get("rows",0),errors="coerce") or 0),"reason":str(r.get("reason","Explicit adapter acquisition status"))})
    if not history.empty:
        for (comp,season,source),g in history.groupby(["competition","season","source_name"],dropna=False):
            for field in CANONICAL_FIELDS[6:]:
                values=g[field] if field in g.columns else pd.Series(pd.NA,index=g.index); nonmissing=int(values.notna().sum()); status="AVAILABLE" if nonmissing==len(g) else "PARTIAL" if nonmissing else "UNAVAILABLE"
                rows.append({"competition":comp,"competition_name":COMPETITION_NAMES.get(comp,str(comp)),"season":season,"source":source,"field":field,"status":status,"fixture_count":int(len(g)),"matched_count":nonmissing,"coverage_rate":float(nonmissing/len(g)) if len(g) else 0.0,"reason":"Observed season/source/field coverage; missing values remain missing"})
    return pd.DataFrame(rows)

def source_reconciliation(history):
    cols=["canonical_key","source_count","sources","duplicate_source_identity","row_count","collision","collision_reason"]
    if history.empty:return pd.DataFrame(columns=cols)
    x=history.copy(); x["canonical_key"]=x["competition"].astype(str)+"|"+x["season"].astype(str)+"|"+x["kickoff_utc"].astype(str)+"|"+x["home_team"].astype(str).str.strip().str.lower()+"|"+x["away_team"].astype(str).str.strip().str.lower(); rows=[]
    for key,g in x.groupby("canonical_key",sort=False):
        sources=sorted(set(g["source_name"].astype(str))); duplicate=bool(len(g)>len(sources)); outcome=[c for c in ("home_goals","away_goals","result") if c in g.columns]; sig=set(tuple(None if pd.isna(r[c]) else str(r[c]) for c in outcome) for _,r in g.iterrows()); collision=bool(len(sources)>1 and len(sig)>1)
        rows.append({"canonical_key":key,"source_count":len(sources),"sources":"|".join(sources),"duplicate_source_identity":duplicate,"row_count":int(len(g)),"collision":collision,"collision_reason":"Conflicting outcome across sources" if collision else ""})
    out=pd.DataFrame(rows,columns=cols); out["duplicate_source_identity"]=out["duplicate_source_identity"].map(lambda x: bool(x)).astype(object); out["collision"]=out["collision"].map(lambda x: bool(x)).astype(object); return out

def run_audit(out_dir="artifacts",config=AuditConfig()):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); history,acquisition=load_available_history(start_year=config.start_year,end_year=config.end_year); fixtures=fixture_audit(history); fields=field_audit(history,config); coverage=coverage_matrix(history,fields,acquisition,config); reconciliation=source_reconciliation(history); pit=pit_audit(fields); plans=pd.DataFrame([{"competition":p.competition,"canonical_candidates":"|".join(p.canonical_candidates),"discovery_only":"|".join(p.discovery_only),"pit_status":p.pit_status,"notes":p.notes} for p in source_plans()]); fixtures.to_csv(out/"fixture_audit.csv",index=False); fields.to_csv(out/"field_audit.csv",index=False); coverage.to_csv(out/"coverage_matrix.csv",index=False); acquisition.to_csv(out/"acquisition_coverage.csv",index=False); reconciliation.to_csv(out/"source_reconciliation.csv",index=False); pit.to_csv(out/"pit_audit.csv",index=False); plans.to_csv(out/"competition_source_plan.csv",index=False); observed=sorted(set(history["competition"].astype(str))) if not history.empty else []; summary={"target_competitions":list(TARGET_COMPETITIONS),"target_competition_count":14,"requested_season_start":config.start_year,"requested_season_end":config.end_year,"observed_competitions":observed,"observed_competition_count":len(observed),"unobserved_competitions":[c for c in TARGET_COMPETITIONS if c not in observed],"fixture_count":int(len(fixtures)),"field_observation_count":int(len(fields)),"pit_status_counts":{str(k):int(v) for k,v in (fields["pit_status"].value_counts().to_dict() if not fields.empty else {}).items()},"duplicate_source_rows":int(reconciliation["duplicate_source_identity"].sum()) if not reconciliation.empty else 0,"collision_rows":int(reconciliation["collision"].sum()) if not reconciliation.empty else 0,"team_mapping_invalid_rows":int((fixtures["team_mapping_status"]=="INVALID").sum()) if not fixtures.empty else 0,"no_missing_to_zero":True,"search_engines_are_not_dataset_sources":True,"audit_scope":"fixture -> season/source/field -> duplicate/collision -> team identity -> PIT publication time","coverage_claim_policy":"observed-only; provider advertisements and search results never become AVAILABLE"}; (out/"audit_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding="utf-8"); return summary
if __name__=="__main__": print(json.dumps(run_audit(),indent=2,ensure_ascii=False,default=str))

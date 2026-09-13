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
    cutoff_minutes=int(config.prediction_cutoff_minutes)
    for _,r in history.iterrows():
        fid=_fixture_id(r); kickoff=pd.to_datetime(r.get("kickoff_utc"),utc=True,errors="coerce"); cutoff=kickoff-pd.Timedelta(minutes=cutoff_minutes) if pd.notna(kickoff) else pd.NaT; available=pd.to_datetime(r.get("source_available_at_utc"),utc=True,errors="coerce"); retrieved=pd.to_datetime(r.get("retrieved_at_utc"),utc=True,errors="coerce")
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

def _season_key(comp,year):
    if comp in {"J1","J2","J3","FRI"}: return str(year)
    return f"{year}/{str(year+1)[-2:]}"

def coverage_matrix(history,field_rows,acquisition=None,config=AuditConfig()):
    rows=[]
    observed=set(history["competition"].dropna().astype(str)) if not history.empty else set()
    for comp in TARGET_COMPETITIONS:
        for year in range(config.start_year,config.end_year+1):
            season=_season_key(comp,year)
            ch=history[(history["competition"].astype(str)==comp) & (history["season"].astype(str)==season)] if not history.empty else pd.DataFrame()
            if len(ch): status="AVAILABLE"; reason="Observed parseable fixture rows from current adapter"
            elif comp=="J3" and year<2014: status="NOT_APPLICABLE"; reason="J3 did not exist in this season cell"
            elif comp not in observed: status="UNAVAILABLE"; reason="No data elsewhere has been established by this adapter audit; this absence is not a claim that no data exists elsewhere."
            else: status="COVERAGE_GAP"; reason="Target season cell has no observed fixture rows from current adapter"
            rows.append({"competition":comp,"competition_name":COMPETITION_NAMES[comp],"season":season,"source":"current_observed_adapter","field":"__FIXTURE_CELL__","status":status,"fixture_count":int(len(ch)),"reason":reason})
    if acquisition is not None and not acquisition.empty:
        for _,r in acquisition.iterrows():
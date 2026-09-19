from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.jleague_adapter import load_jleague_history
from src.data.openfootball_adapter import load_openfootball_history
from src.data.competition_catalog import ACTIVE_SCOPE

LEAGUES={"EPL":"E0","BL1":"D1","SA":"I1","LL":"SP1","FL1":"F1","ERE":"N1"}
BASE="https://www.football-data.co.uk/mmz4281/{season_folder}/{league}.csv"
COMPETITION_TZ={"EPL":"Europe/London","BL1":"Europe/Berlin","SA":"Europe/Rome","LL":"Europe/Madrid","FL1":"Europe/Paris","ERE":"Europe/Amsterdam"}
RAW_STAT_MAP={"home_shots":"HS","away_shots":"AS","home_shots_on_target":"HST","away_shots_on_target":"AST","home_corners":"HC","away_corners":"AC","home_fouls":"HF","away_fouls":"AF","home_yellow_cards":"HY","away_yellow_cards":"AY","home_red_cards":"HR","away_red_cards":"AR"}
HEADERS={"User-Agent":"SoccerPredictionResearch/1.0"}

def _http_get(url: str, *, timeout: int = 45, params: dict | None = None) -> requests.Response:
    """Bounded retrying GET for transient public-data failures.

    Retries are limited to connection errors, timeouts, throttling and 5xx
    responses. A genuine 4xx/404 remains a real data-availability signal and is
    surfaced to the adapter/audit instead of being hidden.
    """
    retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET"}), raise_on_status=False)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    response = session.get(url, params=params, timeout=timeout, headers=HEADERS)
    response.raise_for_status()
    return response

def season_folder(start_year:int)->str:return f"{str(start_year)[-2:]}{str(start_year+1)[-2:]}"
def _parse_football_data_dates(series:pd.Series)->pd.Series:return pd.to_datetime(series.astype("string").str.strip(),format="mixed",dayfirst=True,errors="coerce")
def _parse_kickoff(df:pd.DataFrame,competition:str):
    local_date=_parse_football_data_dates(df["Date"]); time_text=df["Time"].astype("string").str.strip() if "Time" in df.columns else pd.Series(pd.NA,index=df.index,dtype="string")
    has_time=time_text.notna()&time_text.ne("")&time_text.ne("nan"); naive=pd.to_datetime(local_date.dt.strftime("%Y-%m-%d")+" "+time_text,format="%Y-%m-%d %H:%M",errors="coerce"); tz=ZoneInfo(COMPETITION_TZ[competition])
    kickoff=naive.dt.tz_localize(tz,ambiguous="NaT",nonexistent="NaT").dt.tz_convert("UTC"); date_only=local_date.dt.tz_localize(tz,ambiguous="NaT",nonexistent="NaT").dt.tz_convert("UTC"); kickoff=kickoff.where(has_time&kickoff.notna(),date_only)
    precision=pd.Series("DATE_ONLY",index=df.index,dtype="string").mask(has_time&kickoff.notna(),"MINUTE"); return kickoff,precision

def load_season(competition:str,start_year:int,cache_dir:str="data/raw")->pd.DataFrame:
    league=LEAGUES[competition]; url=BASE.format(season_folder=season_folder(start_year),league=league); path=Path(cache_dir)/f"{competition}_{start_year}.csv"; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raw=path.read_bytes()
    else:
        r=_http_get(url,timeout=45); raw=r.content; path.write_bytes(raw)
    df=pd.read_csv(BytesIO(raw)); required={"Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"}; missing=required-set(df.columns)
    if missing: raise ValueError(f"{url}: missing columns {sorted(missing)}")
    kickoff,precision=_parse_kickoff(df,competition)
    out=pd.DataFrame({"match_id":[f"fd:{competition}:{start_year}:{i}" for i in df.index],"competition":competition,"season":f"{start_year}/{str(start_year+1)[-2:]}","season_start":start_year,"kickoff_utc":kickoff,"kickoff_time_available":precision.eq("MINUTE"),"event_time_precision":precision,"source_event_date":_parse_football_data_dates(df["Date"]).dt.date.astype("string"),"home_team":df["HomeTeam"].astype(str).str.strip(),"away_team":df["AwayTeam"].astype(str).str.strip(),"home_goals":pd.to_numeric(df["FTHG"],errors="coerce"),"away_goals":pd.to_numeric(df["FTAG"],errors="coerce"),"result":df["FTR"].astype(str).str.strip(),"source_name":"Football-Data.co.uk","source_record_id":[str(i) for i in df.index],"source_available_at_utc":pd.NaT,"retrieved_at_utc":datetime.now(timezone.utc)})
    for target,source in RAW_STAT_MAP.items(): out[target]=pd.to_numeric(df[source],errors="coerce") if source in df.columns else np.nan
    out["raw_snapshot_id"]=hashlib.sha256(raw).hexdigest(); return out.dropna(subset=["kickoff_utc","home_goals","away_goals"]).reset_index(drop=True)

def _load_one(args):
    comp,year,cache_dir=args
    try:
        d=load_season(comp,year,cache_dir); return comp,year,d,{"competition":comp,"season":year,"status":"AVAILABLE","rows":len(d),"source":"Football-Data.co.uk"}
    except Exception as e:return comp,year,None,{"competition":comp,"season":year,"status":"UNAVAILABLE","rows":0,"source":"Football-Data.co.uk","error":str(e)}

def load_available_history(start_year:int=2010,end_year:int=2025,max_workers:int=8)->tuple[pd.DataFrame,pd.DataFrame]:
    tasks=[(c,y,"data/raw") for c in LEAGUES for y in range(start_year,end_year+1)]
    with ThreadPoolExecutor(max_workers=max(1,min(int(max_workers),len(tasks)))) as pool: results=[f.result() for f in as_completed([pool.submit(_load_one,t) for t in tasks])]
    results.sort(key=lambda x:(list(LEAGUES).index(x[0]),x[1])); primary_frames=[r[2] for r in results if r[2] is not None]; primary_history=pd.concat(primary_frames,ignore_index=True) if primary_frames else pd.DataFrame(); primary_coverage=pd.DataFrame([r[3] for r in results])
    jleague_history,jleague_coverage=load_jleague_history(start_year=start_year,end_year=end_year); cup_history,cup_coverage=load_openfootball_history(start_year=start_year,end_year=end_year,max_workers=max_workers)
    frames=[x for x in (primary_history,jleague_history,cup_history) if not x.empty]; history=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if not history.empty: history=history.sort_values(["competition","kickoff_utc","home_team","away_team","source_name"],kind="mergesort").reset_index(drop=True)
    coverage=pd.concat([primary_coverage,jleague_coverage,cup_coverage],ignore_index=True)\n    if not history.empty:\n        history=history[history["competition"].astype(str).isin(ACTIVE_SCOPE)].copy()\n    if not coverage.empty:\n        coverage=coverage[coverage["competition"].astype(str).isin(ACTIVE_COMPETITION_SET)].copy()\n    return history,coverage

from __future__ import annotations
import json
from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import pandas as pd
from src.data.competition_catalog import ACTIVE_SCOPE, COMPETITION_CATALOG
from src.data.competition_sources import AUXILIARY_COMPETITIONS, AUXILIARY_NAMES, AUXILIARY_PLANS, PLANS

REPO = Path(__file__).resolve().parents[2]
TIMEOUT = 15

FOOTBALL_DATA_CODES = {
    "EPL": "E0", "CHA": "CH", "BL1": "D1", "SA": "I1", "LL": "SP1", "FL1": "F1", "ERE": "N1",
}


def _current_football_data_season_code(now: datetime | None = None) -> str:
    """Return mmz season code for the current football season without hardcoding a year."""
    if now is None:
        now = datetime.now(timezone.utc)
    start_year = now.year if now.month >= 8 else now.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _direct_source_urls(now: datetime | None = None) -> dict[str, str]:
    season = _current_football_data_season_code(now)
    urls = {
        code: f"https://www.football-data.co.uk/mmz4281/{season}/{csv_code}.csv"
        for code, csv_code in FOOTBALL_DATA_CODES.items()
    }
    urls.update({
        "AG_M": "https://www.olympics.com/en/news/football-asian-games-2026-schedule-results-standings-complete-list",
        "AG_W": "https://www.olympics.com/en/news/football-asian-games-2026-schedule-results-standings-complete-list",
        "J1": "https://data.j-league.or.jp/SFMS01/search",
        "J2": "https://data.j-league.or.jp/SFMS01/search",
        "J3": "https://data.j-league.or.jp/SFMS01/search",
        "UCL": "https://raw.githubusercontent.com/openfootball/champions-league/master/2025-26/cl.txt",
        "UEL": "https://raw.githubusercontent.com/openfootball/champions-league/master/2025-26/el.txt",
        "U23_M": "https://www.the-afc.com/en/national/afc_u23_asian_cup.html",
        "U18_M": "https://www.jfa.jp/national_team/u18_2026/",
        "DFBP": "https://raw.githubusercontent.com/openfootball/deutschland/master/2025-26/cup.txt",
        "CAR": "https://raw.githubusercontent.com/openfootball/england/master/2025-26/eflcup.txt",
        "FRI": "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard",
    })
    return urls
ADAPTER_HINTS = {
    "EPL":"football_data","CHA":"football_data","BL1":"football_data","SA":"football_data",
    "LL":"football_data","FL1":"football_data","J1":"jleague","J2":"jleague","J3":"jleague",
    "ERE":"football_data","AG_M":"olympics_results","AG_W":"olympics_results","U23_M":"international_results","U18_M":"international_results",
    "UCL":"openfootball","UEL":"openfootball","DFBP":"openfootball","CAR":"openfootball","FRI":"espn_friendlies",
}
AUX_ADAPTER_HINTS = {
    "WORLD_CUP":"international_adapter","WORLD_CUP_QUALI":"international_adapter",
    "ASIAN_CUP":"international_adapter","EURO":"international_adapter","EURO_QUALI":"international_adapter",
    "NATIONS_LEAGUE":"international_adapter","INTERNATIONAL_FRIENDLY":"international_adapter",
    "EMPERORS_CUP":"jfa_registry_only","INTERHIGH":"jfa_registry_only","JFA_U20":"jfa_registry_only","JFA_U18":"jfa_registry_only",
}

def probe(url):
    if not url:
        return "NOT_PROBED","No explicit repository source URL contract"
    try:
        req=Request(url,headers={"User-Agent":"SoccerPredictionResearch/1.0"})
        with urlopen(req,timeout=TIMEOUT) as r:
            code=int(getattr(r,"status",200))
            return ("REACHABLE" if 200 <= code < 400 else "HTTP_ERROR",str(code))
    except Exception as e:
        return "UNREACHABLE",f"{type(e).__name__}: {e}"[:240]

def observed(code):
    p=REPO/"artifacts"/"acquisition_coverage.csv"
    if not p.exists(): return False
    try:
        d=pd.read_csv(p,usecols=["competition","rows"])
        x=d[d.competition.astype(str)==code]
        return bool(not x.empty and (pd.to_numeric(x.rows,errors="coerce").fillna(0)>0).any())
    except Exception:
        return False

def audit():
    rows=[]
    direct_source_urls = _direct_source_urls()
    for s in COMPETITION_CATALOG:
        if s.code not in ACTIVE_SCOPE:
            continue
        code=s.code
        url=direct_source_urls.get(code,"")
        live,detail=probe(url)
        obs=observed(code)
        adapter=ADAPTER_HINTS.get(code,"")
        if obs: acq="OBSERVED_IN_ARTIFACT"
        elif live=="REACHABLE": acq="LIVE_SOURCE_REACHABLE"
        elif adapter: acq="ADAPTER_PRESENT_NOT_CURRENTLY_PROVEN"
        else: acq="DISCOVERY_REQUIRED"
        rows.append({
            "code":code,"name":s.name,"region":s.region,"tier":s.tier,
            "competition_type":s.competition_type,"universe":"catalog",
            "source_candidates":"|".join(PLANS.get(code,())),
            "adapter_hint":adapter,"artifact_observed":obs,"probe_url":url,
            "live_probe":live,"probe_detail":detail,"acquisition_status":acq,
            "pit_status":"PIT_UNPROVEN","production_eligible":False,
            "production_status":"NOT_ELIGIBLE_UNTIL_PIT_OOS_GATES",
        })
    for code in AUXILIARY_COMPETITIONS:
        if code not in ACTIVE_SCOPE:
            continue
        rows.append({
            "code":code,"name":AUXILIARY_NAMES[code],"region":"Global/Asia/Japan",
            "tier":"auxiliary","competition_type":"auxiliary","universe":"auxiliary",
            "source_candidates":"|".join(AUXILIARY_PLANS.get(code,())),
            "adapter_hint":AUX_ADAPTER_HINTS.get(code,""),"artifact_observed":False,
            "probe_url":"","live_probe":"NOT_PROBED",
            "probe_detail":"Auxiliary source exists in repository; publication-time proof remains separate.",
            "acquisition_status":"ADAPTER_PRESENT_PIT_UNKNOWN" if AUX_ADAPTER_HINTS.get(code)=="international_adapter" else "REGISTRY_ONLY",
            "pit_status":"PIT_UNKNOWN","production_eligible":False,"production_status":"AUXILIARY_ONLY",
        })
    df=pd.DataFrame(rows)
    summary={
        "catalog_entries":len(COMPETITION_CATALOG),
        "active_scope_entries":len(ACTIVE_SCOPE),
        "parked_catalog_entries":len(COMPETITION_CATALOG)-len(ACTIVE_SCOPE),
        "auxiliary_entries":len(AUXILIARY_COMPETITIONS),
        "total_audited":len(df),
        "live_reachable":int((df.live_probe=="REACHABLE").sum()),
        "observed_artifact":int(df.artifact_observed.fillna(False).sum()),
        "discovery_required":int((df.acquisition_status=="DISCOVERY_REQUIRED").sum()),
        "pit_unproven_or_unknown":int(df.pit_status.isin(["PIT_UNPROVEN","PIT_UNKNOWN","UNVERIFIED"]).sum()),
        "production_eligible_granted":0,
    }
    return df,summary

if __name__=="__main__":
    out=REPO/"artifacts"; out.mkdir(exist_ok=True)
    df,summary=audit()
    df.to_csv(out/"full_catalog_audit.csv",index=False)
    (out/"full_catalog_audit_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print(df[["code","name","acquisition_status","live_probe","pit_status"]].to_string(index=False))

# Audit is intentionally rerun on subsequent commits so live source reachability is refreshed.

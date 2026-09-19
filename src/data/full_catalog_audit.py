from __future__ import annotations
import json
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd
from src.data.competition_catalog import COMPETITION_CATALOG
from src.data.competition_sources import AUXILIARY_COMPETITIONS, AUXILIARY_NAMES, AUXILIARY_PLANS, PLANS

REPO = Path(__file__).resolve().parents[2]
TIMEOUT = 15

DIRECT_SOURCE_URLS = {
    "EPL":"https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "CHA":"https://www.football-data.co.uk/mmz4281/2526/CH.csv",
    "BL1":"https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "SA":"https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "LL":"https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "FL1":"https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "J1":"https://www.football-data.co.uk/mmz4281/2526/JPN.csv",
    "J2":"https://www.football-data.co.uk/mmz4281/2526/JPN2.csv",
    "J3":"https://data.j-league.or.jp/",
    "UCL":"https://raw.githubusercontent.com/openfootball/champions-league/master/2025-26/cl.txt",
    "UEL":"https://raw.githubusercontent.com/openfootball/champions-league/master/2025-26/el.txt",
    "DFBP":"https://raw.githubusercontent.com/openfootball/deutschland/master/2025-26/cup.txt",
    "CAR":"https://raw.githubusercontent.com/openfootball/england/master/2025-26/eflcup.txt",
    "FRI":"https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard",
}
ADAPTER_HINTS = {
    "EPL":"football_data","CHA":"football_data","BL1":"football_data","SA":"football_data",
    "LL":"football_data","FL1":"football_data","J1":"jleague","J2":"jleague","J3":"jleague",
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
    for s in COMPETITION_CATALOG:
        code=s.code
        url=DIRECT_SOURCE_URLS.get(code,"")
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

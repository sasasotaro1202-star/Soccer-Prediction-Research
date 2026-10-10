import numpy as np
import pandas as pd
import pytest

from src.prediction.adaptive_router import AdaptiveModelRouter, RouterConfig

MODELS = ("logistic", "logistic_select", "extra_trees", "random_forest", "hist_gb")

def _data(rows=180):
    items=[]
    for i in range(rows):
        ts=pd.Timestamp("2025-01-01",tz="UTC")+pd.Timedelta(days=i)
        y=i%3
        for model in MODELS:
            base={0:[0.50,0.30,0.20],1:[0.25,0.50,0.25],2:[0.20,0.30,0.50]}[y]
            shift={"logistic":0.05,"logistic_select":0.02,"extra_trees":0.00,"random_forest":-0.02,"hist_gb":-0.03}[model]
            p=np.clip(np.asarray(base)+np.asarray([shift,0,-shift]),0.02,0.96); p=p/p.sum()
            items.append({"prediction_time_utc":ts,"y":y,"model":model,"p_home":p[0],"p_draw":p[1],"p_away":p[2],
                          "pit_verified":True,"competition":"EPL","strength_gap_bin":"MID","scoring_environment_bin":"MID",
                          "rest_bin":"NORMAL","starter_status":"ANNOUNCED","odds_missing":"FALSE"})
    return pd.DataFrame(items)

def _ctx():
    return {"competition":"EPL","strength_gap_bin":"MID","scoring_environment_bin":"MID","rest_bin":"NORMAL",
            "starter_status":"ANNOUNCED","odds_missing":"FALSE"}

def test_router_uses_only_pit_verified_history():
    d=_data()
    d.loc[d.index[-1],"pit_verified"]=False
    r=AdaptiveModelRouter(config=RouterConfig(min_context_rows=10))
    r.fit(d,as_of="2025-06-30T00:00:00Z")
    out=r.route(_ctx(),{m:np.array([[1/3,1/3,1/3]]) for m in MODELS})
    assert np.isfinite(out["probabilities"]).all()
    assert np.allclose(out["probabilities"].sum(axis=1),1)

def test_sparse_context_uses_global_fallback():
    r=AdaptiveModelRouter(config=RouterConfig(min_context_rows=500))
    r.fit(_data())
    out=r.route(_ctx(),{m:np.array([[1/3,1/3,1/3]]) for m in MODELS})
    assert out["routing_source"]=="GLOBAL_OOS"

def test_router_is_deterministic():
    r=AdaptiveModelRouter(config=RouterConfig(min_context_rows=10))
    r.fit(_data())
    probs={m:np.array([[0.4,0.3,0.3],[0.2,0.5,0.3]]) for m in MODELS}
    a=r.route(_ctx(),probs); b=r.route(_ctx(),probs)
    assert np.allclose(a["probabilities"],b["probabilities"])

def test_invalid_probabilities_fail_closed():
    r=AdaptiveModelRouter()
    r.fit(_data())
    with pytest.raises(ValueError):
        r.route(_ctx(),{m:np.array([[0.5,0.3,0.2]]) for m in MODELS[:-1]}|{"hist_gb":np.array([[1.1,-0.1,0.0]])})

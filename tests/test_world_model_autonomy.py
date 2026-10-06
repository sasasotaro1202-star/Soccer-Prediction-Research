from pathlib import Path

REPO = Path('.')

def read(path: str) -> str:
    return (REPO / path).read_text(encoding='utf-8')

def test_world_model_lanes_are_present_and_research_only():
    capture = read('.github/workflows/soccer-prospective-inplay-capture.yml')
    maturity = read('.github/workflows/soccer-prospective-inplay-maturity.yml')
    supervisor = read('.github/workflows/soccer-world-model-supervisor.yml')
    assert 'cron: "*/15 * * * *"' in capture
    assert 'contents: write' in capture
    assert 'cron: "7,37 * * * *"' in maturity
    assert 'production_usable' in maturity
    assert 'performance_verified' in maturity
    assert 'cron: "23 */2 * * *"' in supervisor
    assert 'current_main_only' in supervisor
    assert 'gh workflow run' in supervisor

def test_match_state_lane_does_not_consume_market_odds():
    source = read('src/research/prospective_inplay.py')
    policy = read('config/match_state_policy.json')
    assert 'odds_consumed' in source
    assert 'market' not in policy
    assert 'production_usable' in policy
    assert 'auto_adoption' in policy

def test_control_plane_keeps_promotion_outside_supervisor():
    supervisor = read('.github/workflows/soccer-world-model-supervisor.yml')
    assert 'contents: read' in supervisor
    assert 'production_usable":false' in supervisor
    assert 'performance_verified":false' in supervisor
    assert 'promotion_candidate":false' in supervisor
    assert 'gh pr merge' not in supervisor

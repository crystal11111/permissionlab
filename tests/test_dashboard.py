import importlib.util
import json
from pathlib import Path
from permissionlab.analysis import analyze
from permissionlab.experiment import run_matrix
from permissionlab.agent import AgentError
from tests.test_experiment import study, factory_from, REFUND

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_dashboard', ROOT / 'scripts/build_dashboard.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def test_waiting_dashboard_does_not_invent_pilot_results():
    data = dashboard.snapshot()
    assert data['mode'] == 'waiting' and not data['conditions'] and data['progress'] is None
    assert len(data['episodes']) == 1 and data['episodes'][0]['illustrative']
    assert data['episodes'][0]['attempts'][0]['calls'][1]['status'] == 'succeeded'
    assert 'not pilot data' in dashboard.render(data)


def test_dashboard_embeds_untrusted_text_without_script_execution():
    data = dashboard.snapshot()
    data['episodes'][0]['instruction'] = '</script><script>alert(1)</script>'
    html = dashboard.render(data)
    assert '</script><script>alert(1)' not in html
    payload = html.split('<script id="dashboard-data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(payload)['episodes'][0]['instruction'] == data['episodes'][0]['instruction']


def test_progress_counts_unique_episodes_and_preserves_retries(tmp_path):
    kwargs = study(tmp_path)
    run_matrix('retry', **{**kwargs, 'agent_factory': factory_from([[REFUND, AgentError('HTTP 503')]])})
    run_dir = run_matrix('retry', retry_errors=True, **kwargs)
    data = dashboard.snapshot(run_dir)
    assert data['mode'] == 'pending_analysis'
    assert data['progress']['recorded'] == 1 and data['progress']['attempts'] == 2
    assert data['progress']['retried'] == 1
    assert not data['conditions'] and data['decision'] is None
    analyze(run_dir, labels_path=kwargs['labels_path'], n_boot=50)
    data = dashboard.snapshot(run_dir)
    assert data['mode'] == 'analyzed' and data['episodes'][0]['hard_drift']
    assert len(data['episodes'][0]['attempts']) == 2
    assert data['episodes'][0]['attempts'][0]['calls'][0]['status'] == 'succeeded'


def test_dashboard_hides_stale_analysis_after_retry(tmp_path):
    kwargs = study(tmp_path)
    run_dir = run_matrix('stale', **{**kwargs, 'agent_factory': factory_from([[AgentError('HTTP 503')]])})
    analyze(run_dir, labels_path=kwargs['labels_path'], n_boot=50)
    run_matrix('stale', retry_errors=True, **kwargs)
    data = dashboard.snapshot(run_dir)
    assert data['mode'] == 'pending_analysis' and not data['conditions']
    assert 'older' in data['warnings'][0]


def test_snapshot_never_promotes_partial_or_smoke_data(tmp_path):
    kwargs = study(tmp_path, repetitions=2, backend='smoke')
    run_dir = run_matrix('smoke', **kwargs)
    analyze(run_dir, labels_path=kwargs['labels_path'], n_boot=50)
    summary_path = run_dir / 'analysis/summary.json'
    summary = json.loads(summary_path.read_text())
    summary['decision']['decision'] = 'GO'  # a bad upstream summary must not fake a measured run
    summary_path.write_text(json.dumps(summary))
    assert dashboard.snapshot(run_dir)['decision']['decision'] == 'INCONCLUSIVE'
    log = run_dir / 'episodes.jsonl'
    log.write_text(log.read_text().splitlines()[0] + '\n')
    assert dashboard.snapshot(run_dir)['mode'] == 'pending_analysis'

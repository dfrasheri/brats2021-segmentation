"""Experiment tracking. MLflow when it is installed and enabled, no-op otherwise.

Kept optional so the core pipeline and CI never depend on a tracking server.
Enable with `--mlflow` on train.py, or MLFLOW_TRACKING_URI in the environment.
By default runs go to ./mlruns (view with `mlflow ui`).
"""
import os


class _NullRun:
    def log_params(self, params): pass
    def log_metrics(self, metrics, step=None): pass
    def log_artifact(self, path): pass
    def set_tags(self, tags): pass
    def end(self): pass


class _MlflowRun:
    def __init__(self, mlflow, experiment, run_name):
        self.mlflow = mlflow
        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        # MLflow param values are strings capped at 500 chars.
        self.mlflow.log_params({k: str(v)[:500] for k, v in _flatten(params).items()})

    def log_metrics(self, metrics, step=None):
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path):
        if path and os.path.exists(path):
            self.mlflow.log_artifact(str(path))

    def set_tags(self, tags):
        self.mlflow.set_tags(tags)

    def end(self):
        self.mlflow.end_run()


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def start_run(enabled, experiment="brats2021", run_name=None):
    if not enabled:
        return _NullRun()
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; tracking disabled (pip install mlflow)", flush=True)
        return _NullRun()
    return _MlflowRun(mlflow, experiment, run_name)

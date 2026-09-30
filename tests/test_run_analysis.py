import run_analysis


def test_runner_invokes_every_analysis_in_order(monkeypatch) -> None:
    calls = []

    def record_run(command, *, cwd, check):
        calls.append((command[-1], cwd, check))

    monkeypatch.setattr(run_analysis.subprocess, "run", record_run)

    run_analysis.main()

    called_scripts = [
        run_analysis.Path(script_path).name for script_path, _, _ in calls
    ]
    assert called_scripts == list(run_analysis.ANALYSIS_SCRIPTS)
    assert all(
        cwd == run_analysis.Path(run_analysis.__file__).resolve().parent
        for _, cwd, _ in calls
    )
    assert all(check for _, _, check in calls)
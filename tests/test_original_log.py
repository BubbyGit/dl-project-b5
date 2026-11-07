import pytest

from flipbench.original_log import parse_stdout, summarize


def emit_run(original, run, accs):
    for epoch, val_acc in enumerate(accs, 1):
        row = {"epoch": epoch, "train_loss": 1.0 / epoch, "train_acc": 0.5, "val_acc": val_acc, "total_time_seconds": float(epoch)}
        if epoch == 1:
            row["run"] = run
        original.print_training_details(row, is_final_entry=False)
    original.print_training_details({"epoch": "eval", "tta_val_acc": accs[-1] + 0.01, "total_time_seconds": len(accs) + 0.5}, is_final_entry=True)


def test_parses_the_table_printed_by_the_original(original, capsys):
    original.print_columns(original.logging_columns_list, is_head=True)
    emit_run(original, 0, [0.80, 0.90])
    emit_run(original, 1, [0.82, 0.92])
    print("Mean: 0.9250    Std: 0.0141")
    runs = parse_stdout(capsys.readouterr().out)

    assert [run["run"] for run in runs] == [0, 1]
    assert [run["val_acc"] for run in runs] == [0.90, 0.92]
    assert [run["tta_val_acc"] for run in runs] == [0.91, 0.93]
    assert [len(run["epochs"]) for run in runs] == [2, 2]
    assert runs[0]["total_time_seconds"] == 2.5

    summary = summarize(runs)
    assert summary["completed_runs"] == 2
    assert summary["tta_val_acc"]["mean"] == pytest.approx(0.92)
    assert summary["total_time_seconds_first_run"] == 2.5


def test_ignores_an_unfinished_run(original, capsys):
    original.print_columns(original.logging_columns_list, is_head=True)
    emit_run(original, 0, [0.80, 0.90])
    original.print_training_details({"run": 1, "epoch": 1, "train_loss": 1.0, "train_acc": 0.5, "val_acc": 0.8, "total_time_seconds": 1.0}, is_final_entry=False)
    runs = parse_stdout(capsys.readouterr().out)
    assert [run["run"] for run in runs] == [0]

"""Provider tests. All hermetic — no network, no API key."""

import pytest

from quantlab.providers import LocalProvider, ProviderError, get_provider

ROOT = "tests/fixtures"


def test_reads_jsonl():
    bars = LocalProvider(ROOT).bars("SPY")
    assert len(bars) == 260
    assert set(bars[0]) == {"t", "o", "h", "l", "c", "v"}


def test_reads_csv_with_long_field_names():
    bars = LocalProvider(ROOT).bars("QQQ")
    assert len(bars) == 120 and bars[0]["c"] > 0


def test_case_insensitive_symbol():
    assert LocalProvider(ROOT).bars("spy") == LocalProvider(ROOT).bars("SPY")


def test_bars_are_sorted_oldest_first():
    bars = LocalProvider(ROOT).bars("SPY")
    assert all(bars[i]["t"] <= bars[i + 1]["t"] for i in range(len(bars) - 1))


def test_timestamps_are_timezone_aware_utc():
    t = LocalProvider(ROOT).bars("SPY")[0]["t"]
    assert t.tzinfo is not None and t.utcoffset().total_seconds() == 0


def test_limit_keeps_most_recent():
    full = LocalProvider(ROOT).bars("SPY")
    tail = LocalProvider(ROOT).bars("SPY", limit=10)
    assert tail == full[-10:]


def test_unknown_symbol_is_empty_not_an_error():
    """A vendor having nothing is a legitimate answer, not a failure."""
    assert LocalProvider(ROOT).bars("NOSUCH") == []


def test_missing_directory_raises():
    with pytest.raises(ProviderError):
        LocalProvider("does/not/exist").bars("SPY")


def test_malformed_rows_are_skipped(tmp_path):
    f = tmp_path / "BAD.jsonl"
    f.write_text(
        '{"t":"2025-01-01T00:00:00Z","o":1,"h":2,"l":0.5,"c":1.5,"v":10}\n'
        '{"t":"bad","o":"x","h":null,"l":"","c":"z"}\n'
        '{"t":"2025-01-02T00:00:00Z","o":1,"h":2,"l":0.5,"c":1.6,"v":10}\n',
        encoding="utf-8")
    assert len(LocalProvider(str(tmp_path)).bars("BAD")) == 2


def test_get_provider_by_name():
    assert isinstance(get_provider("local", root=ROOT), LocalProvider)


def test_get_provider_rejects_unknown():
    with pytest.raises(ProviderError):
        get_provider("definitely-not-a-provider")

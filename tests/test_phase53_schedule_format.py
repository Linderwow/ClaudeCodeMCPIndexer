"""Phase 53: trigger-aware schedule formatter.

The Phase 50 dashboard only displayed `Repetition.Interval`, so any task
fired by Daily/Weekly/Boot/Logon triggers showed '—' even though it
ran autonomously. User: "MNQ alpha has to have all the retrains and
other stuff run autonomously why do I have these?" — the retrains DID
run; the dashboard hid the schedule.

Phase 53 dumps RAW trigger fields from PowerShell and lets Python
format. These tests pin the formatter so a future PowerShell quirk
doesn't silently regress the schedule rendering.
"""
from __future__ import annotations

from code_rag.dashboard.projects import _format_schedule


def test_empty_returns_none() -> None:
    assert _format_schedule([]) is None
    assert _format_schedule([{}]) is None


def test_repetition_only() -> None:
    """A pure RepetitionPattern with PT5M interval (e.g. health-alert)."""
    out = _format_schedule([
        {"cls": "MSFT_TaskRepetitionPattern", "interval": "PT5M"},
    ])
    assert out == "PT5M"


def test_daily_at_3am() -> None:
    out = _format_schedule([
        {"cls": "MSFT_TaskDailyTrigger",
         "start_boundary": "2026-05-08T03:00:00"},
    ])
    assert out == "daily 03:00"


def test_daily_with_repetition() -> None:
    """e.g. 'daily 03:00, then every 30 min for 24h'."""
    out = _format_schedule([
        {"cls": "MSFT_TaskDailyTrigger",
         "start_boundary": "2026-05-08T03:00:00",
         "interval": "PT30M"},
    ])
    assert out == "daily 03:00 / +PT30M"


def test_weekly_friday() -> None:
    """DaysOfWeek bitfield 32 = Friday only (MSFT enum)."""
    out = _format_schedule([
        {"cls": "MSFT_TaskWeeklyTrigger",
         "start_boundary": "2026-05-08T17:45:00",
         "days_of_week": "32"},
    ])
    assert out == "weekly Fri 17:45"


def test_weekly_weekdays() -> None:
    """Bitfield 62 = Mon|Tue|Wed|Thu|Fri = 'weekdays'."""
    out = _format_schedule([
        {"cls": "MSFT_TaskWeeklyTrigger",
         "start_boundary": "2026-05-08T17:00:00",
         "days_of_week": "62"},
    ])
    assert out == "weekly weekdays 17:00"


def test_weekly_weekends() -> None:
    """Bitfield 65 = Sun|Sat (1 + 64) = 'weekends'."""
    out = _format_schedule([
        {"cls": "MSFT_TaskWeeklyTrigger",
         "start_boundary": "2026-05-08T08:00:00",
         "days_of_week": "65"},
    ])
    assert out == "weekly weekends 08:00"


def test_weekly_string_days() -> None:
    """Some Windows builds return the day-list as a comma-separated
    string instead of the bitfield int."""
    out = _format_schedule([
        {"cls": "MSFT_TaskWeeklyTrigger",
         "start_boundary": "2026-05-08T05:00:00",
         "days_of_week": "Monday, Wednesday"},
    ])
    assert out == "weekly Mon/Wed 05:00"


def test_at_boot() -> None:
    out = _format_schedule([
        {"cls": "MSFT_TaskBootTrigger"},
    ])
    assert out == "at boot"


def test_at_logon() -> None:
    out = _format_schedule([
        {"cls": "MSFT_TaskLogonTrigger"},
    ])
    assert out == "at logon"


def test_monthly() -> None:
    out = _format_schedule([
        {"cls": "MSFT_TaskMonthlyTrigger"},
    ])
    assert out == "monthly"


def test_event_trigger() -> None:
    out = _format_schedule([
        {"cls": "MSFT_TaskEventTrigger"},
    ])
    assert out == "on event"


def test_one_shot_time_trigger() -> None:
    """TimeTrigger without a repetition is a single-fire scheduled time."""
    out = _format_schedule([
        {"cls": "MSFT_TaskTimeTrigger"},
    ])
    assert out == "one-shot"


def test_multi_trigger_combined() -> None:
    """A task with both a Daily AND a separate Weekly trigger renders
    both, comma-separated. Real-world this is rare but possible."""
    out = _format_schedule([
        {"cls": "MSFT_TaskDailyTrigger",
         "start_boundary": "2026-05-08T03:00:00"},
        {"cls": "MSFT_TaskWeeklyTrigger",
         "start_boundary": "2026-05-08T17:00:00",
         "days_of_week": "62"},
    ])
    assert "daily 03:00" in out
    assert "weekly weekdays 17:00" in out
    assert ", " in out


def test_unknown_class_falls_back_to_stripped_name() -> None:
    """A trigger we don't have a special case for renders the cleaned
    class name rather than crashing or showing nothing."""
    out = _format_schedule([
        {"cls": "MSFT_TaskWeirdNewTrigger"},
    ])
    assert out == "WeirdNew"


def test_malformed_start_boundary_doesnt_crash() -> None:
    out = _format_schedule([
        {"cls": "MSFT_TaskDailyTrigger",
         "start_boundary": "garbage-no-T-here"},
    ])
    # No time substring extracted → "daily" with empty time
    assert out == "daily"

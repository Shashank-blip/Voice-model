from datetime import datetime

from jarvis_nlu.skills import clock

NOW = datetime(2026, 8, 24, 14, 5)


def test_get_time_is_conversational_not_iso():
    reply = clock.get_time(NOW)
    assert "2:05" in reply
    assert "14:05" not in reply


def test_get_date_names_weekday_and_month():
    reply = clock.get_date(NOW)
    assert "Monday" in reply
    assert "August" in reply
    assert "24" in reply


def test_midnight_and_noon_read_naturally():
    assert "12:00 AM" in clock.get_time(datetime(2026, 8, 24, 0, 0))
    assert "12:00 PM" in clock.get_time(datetime(2026, 8, 24, 12, 0))

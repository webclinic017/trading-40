from datetime import datetime, timedelta


def num_data_to_start_date(num_data: int, interval: str, end_date: datetime) -> datetime:
    """
    Converts a requested number of data points from before `end_date` to a `start_date`.
    Args:
        num_data:
        interval:
        end_date:

    Returns:

    """
    interval_map = {
        "1m": timedelta(minutes=num_data),
        "1h": timedelta(hours=num_data),
        "1d": timedelta(days=num_data),
    }

    return end_date - interval_map[interval]

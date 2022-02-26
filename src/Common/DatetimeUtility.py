import pandas_market_calendars as mcal
from dateutil import tz
from datetime import timedelta,datetime
import pandas as pd

class DatetimeUtility:
    def __init__(self):
        pass

    def is_market_open_now(self, now=datetime.now(tz=tz.gettz('America/New_York'))):

        from_zone = tz.tzutc()
        to_zone = tz.gettz('America/New_York')

        tomorrow = datetime.utcnow().today() +timedelta(days=1)  
        yesterday = datetime.utcnow().today() + timedelta(days=-1)
    
        #adjust timezone
        tomorrow = tomorrow.replace(tzinfo=from_zone)
        yesterday= yesterday.replace(tzinfo= from_zone)

        # to new york time zone
        tomorrow = tomorrow.astimezone(to_zone)
        yesterday = yesterday.astimezone(to_zone)

        nyse = mcal.get_calendar('NYSE')


        schedule = nyse.schedule(yesterday.date().strftime("%Y-%m-%d"), tomorrow.date().strftime("%Y-%m-%d"))

        is_open=False
        try: 
         is_open=nyse.open_at_time(schedule, pd.Timestamp(now.strftime("%Y-%m-%d %H:%M"), tz='America/New_York'))
        except ValueError:
         is_open = False
        return is_open

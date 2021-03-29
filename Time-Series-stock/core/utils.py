import datetime as dt

class Timer():
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        self.end_dt = dt.datetime.now()
        print('End Time {}'.format(self.end_dt))
        print('Time taken: {}'.format((self.end_dt - self.start_dt)))
        
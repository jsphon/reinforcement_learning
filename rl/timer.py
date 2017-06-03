import time


class Timer(object):
    def __init__(self, name='', output=print):
        self._name = name
        self._output = output

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, a, b, c):
        self.end = time.time()
        self.timetaken = self.end - self.start
        self._output('%s Took %0.2fs seconds' % (self._name, self.timetaken))

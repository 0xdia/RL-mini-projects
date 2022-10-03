
class IncreasingLinearSchedule:
    def __init__(self, start, end, num_steps):
        self.start, self.end = start, end
        self.step = (end - start) / float(num_steps)
        self.current = start

    def __call__(self):
        current_value = self.current
        self.current = self.current + self.step
        if self.current > self.end:
            self.current = self.start
        return current_value
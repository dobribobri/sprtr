

class Log:
    def __init__(self):
        self.last_message = ''
        self.n_repeats = 0

    def print(self, s: str):
        if s == self.last_message:
            self.n_repeats += 1
            print('\r' + s + '\t\tx{}'.format(self.n_repeats),
                  flush=True,
                  end='          ')
        else:
            if self.n_repeats > 0:
                print()
            self.n_repeats = 0
            print(s)
        self.last_message = s

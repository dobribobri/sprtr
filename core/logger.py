import datetime
import traceback


class Log:
    def __init__(self):
        self.last_message = ''
        self.n_repeats = 0

        # with open('log', 'a') as file:
        #     file.write(str(datetime.datetime.now()) + '\n')

    def print(self, s: str):
        with open('log', 'a') as file:
            file.write(str(datetime.datetime.now()) + '\t' + s + '\n')

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

    @staticmethod
    def print_exception(e: Exception):
        with open('log', 'a') as file:
            file.write(str(datetime.datetime.now()) + '\n')
            traceback.print_exception(type(e), value=e, tb=e.__traceback__, file=file)

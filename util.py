import json
import subprocess
from time import time


def lines_in_file(fname):
    """Returns the number of lines in fname"""
    return int(subprocess.check_output(
        ['wc', '-l', fname]).strip().split()[0])


def load_json(fname):
    """Reads the JSON data in fname and returns it as a dictionary"""
    with open(fname) as f:
        return json.loads(f.read())


def write_json(d, fname):
    """Writes dictionary d to fname"""
    with open(fname, 'w') as f:
        f.write(json.dumps(d))


def load_json_lines(fname):
    """Yields the JSON data in fname, which should have one JSON object per line"""
    with open(fname) as f:
        for line in f:
            yield json.loads(line)


class LoopLogger():
    """Class for printing out the progress of iteration"""
    def __init__(self, step_size, size=0, print_time=False):
        self.step_size = step_size
        self.size = size
        self.n = 0
        self.print_time = print_time

    def step(self):
        if self.n == 0:
            self.start_time = time()

        self.n += 1
        if self.n % self.step_size == 0:
            if self.size == 0:
                print 'On item ' + str(self.n)
            else:
                print '{:}/{:}, {:.1f}%,'.format(self.n, self.size,
                                                 100.0 * self.n / self.size),
                if self.print_time:
                    time_elapsed = time() - self.start_time
                    print "elapsed: {:.1f}s,".format(time_elapsed),
                    time_per_step = time_elapsed / self.n
                    print "remaining: {:.1f}s".format((self.size - self.n)
                                                      * time_per_step)


def logged_loop(iterable, loop_logger):
    """Iterate through iterable while printing out the progress with oop_logger"""
    loop_logger.n = 0
    for elem in iterable:
        loop_logger.step()
        yield elem

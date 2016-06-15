#!/usr/bin/env python
""" A series of classes that specify various step size functions"""

import math

class PowerSteps():
    """Generate a sequence of steps which consist of a points distributed by a
    power law which contain a number of linear steps between each jump.
    :param num_linear The number of linear steps between each jump in power. The
    distance between these steps is going to increase as the gap between powers
    become larger. In the case when there num_linear is greater than the number
    of steps between powers every step is returned.
    :param pow_jump The jumps in power

    Given num_linear = 9 and pow_jump = 1 this will produce the series
    1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 ...
    """
    def __init__(self, num_linear=9, pow_jump=1):
        self.num_linear = num_linear
        self.pow_jump = pow_jump
        self.power = 0
        self.curr_step = 0
        self.delta_step = self.get_delta_step()

    def next(self):
        """ Calculate the next value in the series"""
        self.curr_step = int(self.curr_step + self.delta_step)
        if self.curr_step >= math.pow(10, self.power+1):
            self.power += 1
            self.curr_step = int(math.pow(10, self.power))
            self.delta_step = self.get_delta_step()
        return self.curr_step

    def __next__(self):
        return self.next()

    def get_delta_step(self):
        """ Calculate the step size"""
        delta_s = (math.pow(10, self.power+1) - math.pow(10, self.power))
        delta_s /= self.num_linear
        if delta_s < 1:
            delta_s = 1
        return delta_s


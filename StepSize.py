#!/usr/bin/env python
""" A series of classes that specify various step size functions"""

import math

class PowerSteps(object):
    """Generate a sequence of steps with a power law

    A sequence of steps which consisting of points spaced by a
    power law which contain a number of linear steps between each jump.

    Args:
        num_linear (int): The number of linear steps between each jump in power.
            The distance between these steps is going to increase as the gap
            between powers become larger. In the case when there num_linear is
            greater than the number of steps between powers every step is
            returned.
        pow_jump (int): The jumps in powers of 10

    Example:

        >>> p = PowerSteps(9,1)
        >>> [ p.next() for i in range(19) ]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    """
    def __init__(self, num_linear=9, pow_jump=1, start=0):
        self.num_linear = num_linear
        self.pow_jump = pow_jump
        self.power = 0
        self.curr_step = 0
        self.delta_step = self.get_delta_step()
        self.start = start

    def next(self):
        """ Calculate the next value in the series"""
        self.curr_step = int(self.curr_step + self.delta_step)
        if self.curr_step >= (self.num_linear+1)*math.pow(10, self.power):
            self.power += 1
            self.curr_step = (self.num_linear+1)*int(math.pow(10, self.power-1))
            self.delta_step = self.get_delta_step()
        return self.curr_step + self.start

    def __next__(self):
        return self.next()

    def get_delta_step(self):
        """ Calculate the step size"""
        delta_s = (math.pow(10, self.power))
        if delta_s < 1:
            delta_s = 1
        return delta_s

    def __lt__(self, other):
        return self.get_delta_step() < other.get_delta_step()

#!/usr/bin/python
# cython: profile=True

"""
Slice sampler for hyperparameters and other variables.

Log version based on Iain Murray's matlab code:
    http://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/slice_sample.m

"""

from __future__ import division

import random # TODO change to gsl_rng, pass in rng to funcs.
# DONE: using rng_container passed to log_slice_sampler

#from math import exp, sqrt, pi, log
from libc.math cimport log, exp, sqrt, M_PI
import numpy
cimport numpy

import sys # just for printing
PRINTING = False
#PRINTING_LOTS = True #False
PRINTING_LOTS = False

MAX_STEPS_OUT = 100 # maximum number of steps to take in stepout_interval_rng


# python/numpy type in numpy arrays
DTYPE = numpy.float64 # this is a c *double*
# ctype
ctypedef numpy.float64_t DTYPE_t

cpdef list log_slice_sampler(log_prob_f, DTYPE_t x, DTYPE_t step,
                             rng_container, long num_samples=1,
                            x_min=None, x_max=None):
    """Samples log_prob_f starting from x; returns num_samples sampled values.
    Step is a parameter used for stepping out (w).
    x_min and x_max are bounds on possible values of x.
    """
    samples = []

    cdef DTYPE_t lpx, log_u, xl, xr, x_new, lpx_new

    for i in xrange(num_samples):

        lpx = log_prob_f(x) # fx returns negative/log value
        #log_u = log(random.random()) + lpx # log(rand*px)
        # Iain Murray's version
        #log_u = log(rng_container.rng_random()) + lpx
        #from R. Neal: "z = log(y) = g(x0 ) − e, where e is exponentially
        #distributed with mean one" (e = -2log(uniform); exp(0.5))
        log_u = lpx + (2 * log(rng_container.rng_random()))
        if PRINTING: print 'START AT: x', x, 'lpx', lpx, 'u', log_u
        sys.stdout.flush()
        xl, xr = stepout_interval_rng(log_prob_f, x, log_u, step,
                                      rng_container, x_min, x_max) # same as non-log
        if PRINTING_LOTS: print 'START AT: x', x, 'lpx', lpx, 'u', log_u, 'xl', xl, 'xr', xr
        sys.stdout.flush()
        loops = 0
        while True:
            #x_new = random.uniform(xl, xr)
            # xprime(dd) = rand()*(x_r(dd) - x_l(dd)) + x_l(dd);
            x_new = (rng_container.rng_random() * (xr - xl)) + xl
            lpx_new = log_prob_f(x_new)
            if PRINTING_LOTS: print 'LOOP: xl', xl, 'xr', xr, 'x_new', x_new, 'lpx', lpx_new, 'u', log_u
            sys.stdout.flush()
            if lpx_new > log_u:
                break
            else:
                xl, xr = log_shrink_interval(x_new, x, xl, xr)
            loops += 1

        if PRINTING: print 'SAMPLED %f; looped %d times' % (x_new, loops)
        sys.stdout.flush()
        samples.append(x_new)
        x = x_new

    return samples

cdef tuple stepout_interval_rng(function, DTYPE_t x, DTYPE_t u, DTYPE_t step,
                            rng_container, x_min=None, x_max=None):
    """ Finds xl and xr such that f(xl) < u and f(xr) < u.
        f(x) nor x are not necessarily bounded at 0/1, eg hyperparameters.
    Need to depend on function to be floor/ceilinged correctly.
        Stops if more than max steps/adjustments have been made; this
    *shouldn't* happen but sometimes occurs in tiny test datasets. """
    cdef DTYPE_t r, xl, xr
    adjustments_l = 0
    adjustments_r = 0
    #r = random.random()
    r = rng_container.rng_random()
    xl = x - (r * step) #max(x - (r*step), VERY_SMALL)
    xr = x + ((1-r) * step) #min(x + ((1 - r)*step), 1)

    if x_min and xl < x_min:
        xl = x_min
    if x_max and xr > x_max:
        xr = x_max

    if PRINTING_LOTS: print 'Orig', xl, xr, 'u', u, 'r', r
    #if PRINTING_LOTS: print 'f(xl)', xl, function(xl)
    #if PRINTING_LOTS: print 'f(xr)', xr, function(xr)

    while function(xl) > u and adjustments_l < MAX_STEPS_OUT:
        if PRINTING_LOTS: print 'Adjusting xl', xl, function(xl), u
        xl = xl - step
        adjustments_l += 1
        if x_min and xl < x_min:
            xl = x_min
            break
    while function(xr) > u and adjustments_r < MAX_STEPS_OUT:
        if PRINTING_LOTS: print 'Adjusting xr', xr, function(xr), u
        xr = xr + step
        adjustments_r += 1
        if x_max and xr > x_max:
            xr = x_max
            break
        if PRINTING_LOTS: print 'Adjusted xr', xr, function(xr), u
    if PRINTING: print 'Adjusted %d times, left %d, right %d' % (
        adjustments_l + adjustments_r, adjustments_l, adjustments_r)
    if adjustments_l > MAX_STEPS_OUT:
        print "Adjusted left more than %d times!" % MAX_STEPS_OUT
    if adjustments_r > MAX_STEPS_OUT:
        print "Adjusted right more than %d times!" % MAX_STEPS_OUT
    return xl, xr

cdef tuple log_shrink_interval(DTYPE_t x_new, DTYPE_t x_orig, DTYPE_t xl,
                               DTYPE_t xr):
    if x_new > x_orig:
        xr = x_new
    elif x_new < x_orig: # NEW for log
        xl = x_new
    return xl, xr


""" Version without log-function."""

def slice_sampler(function, x, step, num_samples=1):
    """Samples function starting from x; returns num_samples sampled values.
    Step is a parameter used for stepping out (w).
    TODO: add rng when cythonising.
    """
    samples = []

    for i in range(num_samples):

        fx = function(x) # what if fx returns negative/log value?
        u = random.uniform(0, fx)
        xl, xr = stepout_interval(function, x, u, step)
        if PRINTING: print 'START AT: x', x, 'fx', fx, 'u', u, 'xl', xl, 'xr', xr

        while True:
            x_new = random.uniform(xl, xr)
            fx_new = function(x_new)
            if PRINTING: print 'LOOP: xl', xl, 'xr', xr, 'x_new', x_new, 'fx', fx_new, 'u', u
            if fx_new > u:
                break
            else:
                xl, xr = shrink_interval(x_new, x, xl, xr)

        if PRINTING: print 'SAMPLED', x_new
        samples.append(x_new)
        x = x_new

    return samples


cdef tuple stepout_interval(function, DTYPE_t x, DTYPE_t u, DTYPE_t step):
    """function/x is not necessarily bounded at 0/1, eg hyperparameters.
    Need to depend on function to be floor/ceilinged correctly."""
    cdef DTYPE_t r, xl, xr
    r = random.random()
    xl = x - r*step #max(x - (r*step), VERY_SMALL)
    xr = x + ((1-r)*step) #min(x + ((1 - r)*step), 1)
    if PRINTING: print 'Orig', xl, xr, 'u', u
    while function(xl) > u:
        if PRINTING: print 'Adjusting xl', xl, function(xl)
        xl = xl - step
        #if xl <= VERY_SMALL: xl = VERY_SMALL break
    while function(xr) > u:
        if PRINTING: print 'Adjusting xr', xr, function(xr)
        xr = xr + step
        #if xr >= 1.0: xr = 1.0 break
    return xl, xr

def shrink_interval(x_new, x_orig, xl, xr):
    if x_new > x_orig:
        xr = x_new
    else:
        xl = x_new
    return xl, xr


def fn_exp(x, l):
    fx = 0
    if x > 0:
        fx = l * exp(-l * x)
    return fx

def fn_mackay(x):
    fx = 0
    if x > 0 and x < 10:
        if x < 1:
            fx = 10
        else:
            fx = 1
    return fx

def main():
    # normal distribution w mu 0 and sigma 1
    #f_norm = lambda x: exp(-x**2/2)/sqrt(2*pi)
    #samples = slice_sampler(f_norm, 0.5, 0.2, 100)

    # exponential w/ lambda = 1.5
    #f_exp = lambda x: fn_exp(x, 1.5)
    #samples = slice_sampler(f_exp, 0.5, 0.2, 100)

    # mackay's skewed density in fig 29.17
    #f_mackay = lambda x: fn_mackay(x)
    #samples = slice_sampler(f_mackay, 0.5, 0.2, 100)

    f_log = lambda x: log(x)
    samples = log_slice_sampler(f_log, 0.5, 0.2, 100000)

    ss = sorted(samples)
#   for s in ss:
#       print s, 1, 2 #, f_exp(s)

if __name__ == "__main__":
    main()


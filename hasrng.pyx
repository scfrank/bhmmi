#!/usr/bin/python
### cython: profile=True

from __future__ import division

cimport numpy
import numpy
import time

from libc.math cimport exp

# python/numpy type in numpy arrays
DTYPE = numpy.float64 # this is a c *double*
# ctype
ctypedef numpy.float64_t DTYPE_t

DEF DRAW_BEST = False #True

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    double gsl_rng_uniform (gsl_rng *r)
    long gsl_rng_uniform_int (gsl_rng *r, long n)
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type *T)
    void gsl_rng_set (gsl_rng * r, unsigned long int s)

cdef long c_random_uniform_int(gsl_rng* rng, long max):
    return gsl_rng_uniform_int(rng, max)

cdef double c_random(gsl_rng* rng):
    return gsl_rng_uniform(rng)

""" Functions for random weighted drawing. """

cdef long draw_c(gsl_rng *rng, list ps, double total):
    cdef double r = gsl_rng_uniform(rng) * total
    cdef int i = 0
    cdef double p
    #while True: # achtung baby
    for p in ps:
        #p = ps[i]
        r = r - p
        if r < 0:
            return i
        i = i + 1

cdef long log_draw_c(gsl_rng *rng, DTYPE_t[:] ps, DTYPE_t max_weight) except? -1:
    cdef DTYPE_t[:] exp_p = ps
    cdef DTYPE_t total_p = 0
    cdef DTYPE_t ep = 0
    cdef long j = 0
    for j in range(len(ps)):
        ep =  exp(ps[j] - max_weight)
        exp_p[j] = ep
        total_p += ep
#    print exp_p, len(exp_p), len(ps)
    cdef double r = gsl_rng_uniform(rng) * total_p
    cdef long i = 0
    while True: # achtung baby
        r = r - exp_p[i]
        if r < 0:
            return i
        i = i + 1

cdef long log_draw_c_list(gsl_rng *rng, list ps, DTYPE_t max_weight) except? -1:
    cdef list exp_p = ps
    cdef DTYPE_t total_p = 0
    cdef DTYPE_t ep = 0
    cdef long j = 0
    for j in range(len(ps)):
        ep =  exp(ps[j] - max_weight)
        exp_p[j] = ep
        total_p += ep
    cdef double r = gsl_rng_uniform(rng) * total_p
    cdef long i = 0
    while True: # achtung baby
        r = r - exp_p[i]
        if r < 0:
            return i
        i = i + 1


cdef list draw_n_samples(gsl_rng *rng, dict counts_dict, double count_total, int num_samples,
                           int K, DTYPE_t alpha_c, int next_id): #, list keys):
    cdef list samples = [0] * num_samples
    cdef int s,i

    cdef list counts = counts_dict.values()
    cdef list keys = counts_dict.keys()
    counts.append(alpha_c) # using original category counts, not counts+alpha
    count_total = count_total + alpha_c
    for i in range(num_samples):
        if DRAW_BEST:
            s = draw_best(numpy.array(counts)) # but counts isn't a probdist? #
        else:
            s = draw_c(rng, counts, count_total)
        if s == len(counts)-1 and i < num_samples: # add a new new category for next time
            #counts = numpy.append(counts, alpha_c)
            counts.append(alpha_c)
            counts[s] = counts[s] - alpha_c # remove hyperparam count
            count_total = count_total - alpha_c

        if s >= K: # new categories get vapid new ids
            # this generates new ids that are 1 greater than next_cat_id
            samples[i] = next_id - K + s
        else:
            #samples.append(keys[s]) # samples must be cat ids
            samples[i] = keys[s]
    return samples

cdef tuple draw_n_samples_with_indices(gsl_rng *rng, int num_samples, list counts, list keys,
                                       DTYPE_t counts_total,
                                       DTYPE_t alpha_c, int next_id):
    """Returns dict of category assignments -> vowels for each category."""
    cdef list samples = [0] * num_samples
    cdef dict sample_indices = {}

    cdef int s
    cdef int K = len(counts)
    cdef int new_cat_id
#    cdef list counts = counts_dict.values()
#    cdef list keys = counts_dict.keys()
    counts.append(alpha_c) # using original category counts, not counts+alpha
    counts_total += alpha_c
    for i in range(num_samples): # is also index into vowel_data
        if DRAW_BEST:
            s = draw_best(numpy.array(counts)) # but counts isn't a probdist...
        else:
            s = draw_c(rng, counts, counts_total)
        #print 'at i', i, 'drew s', s, 'K', K, counts, keys
        if s == len(counts)-1 and i < num_samples: # add a new new category for next time
            counts.append(alpha_c)
            counts[s] -= alpha_c # remove hyperparam count
            counts_total -= alpha_c

        if s >= K: # new categories get vapid new ids
            # this generates new ids that are 1 greater than next_cat_id
            new_cat_id = next_id - K + s
        else:
            #samples.append(keys[s]) # samples must be cat ids
            new_cat_id = keys[s]
        samples[i] = new_cat_id
        if new_cat_id in sample_indices:
            sample_indices[new_cat_id].append(i)
        else:
            sample_indices[new_cat_id] = [i]
    return (samples, sample_indices)

""" FOR DEBUGGING PURPOSES ONLY"""
cdef long draw_best(DTYPE_t[:] ps) except? -1:
    return numpy.argmax(ps)


cdef class HasRng: # cdef is necessary for gsl_rng

    cdef gsl_rng* rng
    cdef long seed

    def __init__(self):
        """Setting rng has to happen elsewhere (ie in a cdef function)."""
        """If it does not happen, this will lead to a segmentation fault."""
        pass

    def set_rng_with_seed(self, randomseed):
        self._set_rng_with_seed(randomseed)
        self.seed = randomseed

    cdef void _set_rng_with_seed(self, long randomseed):
        if randomseed >= 0:
            seed = randomseed
        else:
            seed = int(time.time())
        print "HasRng: Setting random seed to %d" % seed
        self.rng = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self.rng, seed)
        self.seed = seed

    def set_rng_with_rng(self, container):
        """ For calling from python """
        self._set_rng_with_rng(container)
        self.seed = container.get_seed()

    cdef void _set_rng_with_rng(self, HasRng container):
        #gsl_rng* in_rng = container.rng
        self.rng = container.rng # in_rng

    def get_seed(self):
        return self.seed

    """Calling c functions here, since self.rng isn't available to Python
    code."""
    # TODO: how much does this intermediate function call slow us down?

    def rng_random_uniform_int(self, long max):
        """Uniform distribution over integers.
        (changed name to rng_random_uniform_int; was (lexdistsem code) rng_random_uniform)"""
        return c_random_uniform_int(self.rng, max)

    def rng_random(self):
        """Uniform distribution over floats."""
        return c_random(self.rng)

    def rng_draw(self, list probs, DTYPE_t total):
        return draw_c(self.rng, probs, total)

    def rng_log_draw(self, DTYPE_t[:] probs, DTYPE_t max_prob):
        return log_draw_c(self.rng, probs, max_prob)

    def rng_log_draw_list(self, list probs, DTYPE_t max_prob):
        return log_draw_c_list(self.rng, probs, max_prob)

    def rng_draw_n_samples(self, dict counts_dict, double count_total, int num_samples,
                             int K, DTYPE_t alpha_c, int next_id):
        return draw_n_samples(self.rng, counts_dict, count_total, num_samples,
                              K, alpha_c, next_id)

    def rng_draw_n_samples_with_indices(self, int num_samples, list counts, list keys,
                                       DTYPE_t counts_total,
                                       DTYPE_t alpha_c, int next_id):
        return draw_n_samples_with_indices(self.rng, num_samples, counts, keys,
                                       counts_total, alpha_c, next_id)


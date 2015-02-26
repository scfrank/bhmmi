#!/usr/bin/python
## cython: profile=True

"""
To compile:
    python setup_BhmmI.py build_ext --inplace
To run:
    time python run_bhmmi.py
"""

from __future__ import division

import argparse
import numpy as np
import scipy
import math
import sys

from scipy.stats.distributions import entropy

cimport cython
cimport numpy as np
from libc.math cimport log, exp, pow, lgamma

import slice_sampler
from hasrng import HasRng

# GLOBALS

# Double/Float type
# python/numpy type in numpy arrays
DTYPE = np.float64
# ctype
ctypedef np.float64_t DTYPE_t

# Count type (int)
# python/numpy type in numpy arrays
CTYPE = np.int64
# ctype
ctypedef np.int64_t CTYPE_t

# DEF: compile time constants
DEF VERY_SMALL = 1e-100 # smallest HP value permitted

# set to True (& recompile) if debugging: calls check_counts()
# (slows things down a lot otherwise)
DEF CHECK_COUNTS = False

"""
BHMM-I model from:
Adding sentence types to a model of syntactic category
acquisition. Stella Frank, Sharon Goldwater, Frank Keller. TopiCS in Cognitive
Science 5 (3), pp. 495--521. 2013.

The number of inferred tags (K) is set to be the number of true tags in
the dataset. (Also note that this includes the boundary tag (XXX)!)

Note that estimating hyperparameters causes beta to become tag-specific, i.e.
BhmmI.beta is a vector over distinct betas for each tag's emission
distribution. When hyperparameters are not estimated, the beta vector has the
same value in each cell.

Missing ability to do semi-supervised/tag dictionary based learning.

Conditional S-type is *always* i-1 (even in trigrams)


TODO:
    input in treebank format (would be nice to have)
"""

class BHMMI:

    def __init__(self, words, stypes, K, V, alpha, beta, randomseed):
        """
            words: vector of words/observations, including btw-sentence
                markers (XXX/xxx).
            stypes: vector of sentence types, matched to words.
            K: number of latent states
            V: size of vocabulary
            alpha: initial dirichlet parameter for transitions
            beta: inititial dirichlet parameter for emissions
            randomseed: initialises a random number generator (rng) that is
                shared throughout (also passed to slice sampler etc). This
                allows for debugging and replication if seed is set.
        """
        assert (len(words) == len(stypes))
        # make sure the first lines have id 0 for boundary
        assert words[0] == 0
        assert stypes[0] == 0

        self.N = len(words) # Number of word tokens
        self.X = np.array(words,  dtype=CTYPE) # Array of words (ids)
        self.Y = np.zeros(self.N, dtype=CTYPE) # Array of tags (ids)
        self.S = np.array(stypes, dtype=CTYPE) # Array of stypes, co-indexed
                                             # with words and tags
        self.D = len(set(stypes)) # Number of types of sentences
        self.K = K  # Number of types of tags (including XXX/boundary)
        # V must be at least as large as input observations
        assert (V >= len(set(self.X)))
        self.V = V # Sie of vocabulary (number of types of words)
        self.alpha = alpha #transition hyperparameter

        # emission hyperparameter beta is a vector of K values, one for each
        # tag category. If HPs are not sampled, beta will stay symmetrical.
        self.beta = np.array([beta] * K, dtype=DTYPE)

        # Count matricies
        # tag x word matrix of emission probabilities
        self.emissions = np.zeros((self.K, self.V), dtype=CTYPE)
        # y_sum tracks how many times each tag has been used (this avoids an expensive
        # sum operation in prob_k)
        self.y_sums = np.zeros(self.K, dtype=CTYPE)
        # stype x tag x tag matrix of transition probabilities
        self.bi_trans = np.zeros((self.D, self.K, self.K), dtype=CTYPE)
        #trigram counts: Initialised, but not updated (do not use if bigram)
        # stype x tag x tag x tag matrix of transition probabilities
        self.tri_trans = np.zeros((self.D, self.K, self.K, self.K),
                dtype=CTYPE)

        # Random number generator is bound to this BHMM_I instance. Will be
        # passed to slice sampler as well.
        self.rng = HasRng()
        if randomseed:
            print 'Setting rng w', randomseed
            self.rng.set_rng_with_seed(randomseed)
        else:
            self.rng.set_rng_with_seed(-1)

    def initialise(self, init="random", trigram=False, truetags=False):
        """
            Initialises the tags (self.Y) and collects counts.
            Initialisation options are "random", "true", "single" (1 category)
        """
        self.trigram = trigram # boolean whether to use trigram model

        if init == "true" or init == "gold":
            assert truetags, "Missing input true tags for gold initialisation!"
        self.truetags = truetags

        for i in range(self.N):
            if self.X[i] > 0: # not boundary
                if init == "true" or init == "gold":
                    new_y = truetags[i]
                if init == "random":
                    new_y = self.rng.rng_random_uniform_int(self.K-1) + 1
                if init == "single":
                    new_y = 1
                self.Y[i] = new_y

            self.emissions[self.Y[i], self.X[i]] += 1
            self.y_sums[self.Y[i]] += 1
            if 0 < i:
                self.bi_trans[self.S[i-1], self.Y[i-1], self.Y[i]] +=1
            if 1 < i and self.trigram:
                self.tri_trans[self.S[i-1], self.Y[i-2], self.Y[i-1], self.Y[i]] +=1
        if CHECK_COUNTS: self.check_counts()

    def sample(self, num_iters=10, anneal=False, samplehyperparameters=True):
        print "SAMPLING"

        if anneal:
            if num_iters < 20:
                print "Not enough iterations to anneal: not annealling, T=1"
                anneal = False
            else:
                # Annealling schedule from original code; somewhat arbitrary.
                annealling_schedule = [1.0] + [1/(1.03**i) for i in range(19)]
                annealling_schedule.reverse()

        temp = 1
        for iteration in range(num_iters):
            if (iteration % 10) == 0:
                em_probs, trans_probs = self.log_prob()
                print "iter: %d LP: %.2f Em: %.2f Trans: %.2f  VM: %.4f" % (
                        iteration, em_probs + trans_probs, em_probs, trans_probs,
                        self.evaluate_vm())
            if anneal and iteration % (num_iters//20) == 0:
                if iteration < (num_iters//20)*20:
                    temp = annealling_schedule[iteration // (num_iters//20)]
                    print "iter %d: Changing temperature to %f" % (iteration, temp)
            if samplehyperparameters and (iteration % 10) == 0 and iteration >= 10:
                print "iter %d: Sampling Hyperparameters" % iteration
                self.sample_hyperparameters()
            for i in range(self.N):
                if self.X[i] > 0: # not boundary (don't resample boundaries)
                    self.decrement(i)
                    new_y = self.sample_y(i, temp)
                    self.Y[i] = new_y
                    self.increment(i)
                    #print i, self.X[i], self.Y[i]
                    #self.print_state()
                    if CHECK_COUNTS: self.check_counts()
        print "FINISHED"
        em_probs, trans_probs = self.log_prob()
        print "Final sample: LP: %.2f Em: %.2f Trans: %.2f  VM %.4f" % (em_probs
                        + trans_probs, em_probs, trans_probs,
                        self.evaluate_vm())
        if CHECK_COUNTS: self.check_counts()
        #self.print_state()

    def sample_y(self, i, temp=1):
        """ If underflow is a problem we need to sample in log-space. (So far it isn't.)"""

        cdef DTYPE_t p
        cdef CTYPE_t y, new_y

        # cython speed stuff - passed on to prob_k
        # local definitions of things that don't change speed things up
        temp_ems = self.emissions
        temp_bi_trans = self.bi_trans
        temp_tri_trans = self.tri_trans
        cdef np.ndarray[CTYPE_t, ndim=1] y_sums = self.y_sums
        cdef CTYPE_t K = self.K
        cdef CTYPE_t V = self.V
        cdef DTYPE_t alpha = self.alpha
        cdef CTYPE_t x = self.X[i]

        # Conditioning variables: pass these through from sample_y
        cdef CTYPE_t sm1 = self.S[i-1]
        cdef CTYPE_t si = self.S[i]
        cdef CTYPE_t sp1 = self.S[i+1]
        cdef CTYPE_t ym2 = self.Y[i-2]
        cdef CTYPE_t ym1 = self.Y[i-1]
        cdef CTYPE_t yp1 = self.Y[i+1]
        cdef CTYPE_t yp2 = self.Y[i+2]

        cdef list probs = [0.0] * K # value for y=Boundary
        cdef DTYPE_t sum_probs = 0
        cdef DTYPE_t ysum

        for y in range(y_sums.shape[0]):
            ysum = y_sums[y]
            if y > 0:
                p = self.prob_k(
                        i, y, x,
                        temp_ems,
                        ysum,
                        temp_bi_trans,
                        temp_tri_trans,
                        self.trigram,
                        K, V, alpha,
                        self.beta[y],
                        sm1, si, sp1, ym2, ym1, yp1, yp2
                        )
                p = pow(p, temp)
                probs[y] = p
                sum_probs += p
                assert p > 0, "Zero prob/underflow? at i: %d, tag y %d,\
                                      prob %f, %f" % (i, y, p, probs[y])

        # rng_draw does not have to be normalised.
        new_y = self.rng.rng_draw(probs, sum_probs)
        return new_y

    def prob_k(self, int i, int y, int x,
            np.ndarray[CTYPE_t, ndim=2] temp_ems,
            CTYPE_t temp_y_sum,
            np.ndarray[CTYPE_t, ndim=3] temp_bi_trans,
            np.ndarray[CTYPE_t, ndim=4] temp_tri_trans,
            CTYPE_t trigram_bool,
            CTYPE_t K, CTYPE_t V, DTYPE_t alpha, DTYPE_t beta,
            CTYPE_t sm1, CTYPE_t si, CTYPE_t sp1, CTYPE_t ym2, CTYPE_t ym1, CTYPE_t yp1, CTYPE_t yp2
            ):

        cdef DTYPE_t prob = 1

        # These may be useful for printing out components of prob
        #cdef DTYPE_t prob_trans_in
        #cdef DTYPE_t prob_trans_out1
        #cdef DTYPE_t prob_trans_out2
        #cdef DTYPE_t prob_em

        cdef CTYPE_t extras1, extras2

        cdef CTYPE_t eq01 = ym2 == ym1
        cdef CTYPE_t eq12 = ym1 == y
        cdef CTYPE_t eq23 = y == yp1
        cdef CTYPE_t eq34 = yp1 == yp2
        cdef CTYPE_t eq02 = y == ym2
        cdef CTYPE_t eq24 = y == yp2
        cdef CTYPE_t eq13 = ym1 == yp1

        # P(x|y)
        prob *= dir_prob(temp_ems[y, x], temp_y_sum, V, beta)
        #print "prob_em", prob_em
        #prob *= prob_em

        # P(y | y-1, y-2)
        if trigram_bool:
            prob *= dir_prob(
                        temp_tri_trans[sm1, ym2, ym1, y],
                        temp_bi_trans[sm1, ym1, y],
                        K, alpha)
            #print "prob_trans_in", prob_trans_in
            #prob *= prob_trans_in
            prob *= dir_prob(
                        temp_tri_trans[si, ym1, y, yp1]
                        + (eq01 & eq12 & eq23), # extras3
                        temp_bi_trans[si, y, yp1]
                        + (eq01 & eq12), # extras0
                        K, alpha)
            if (yp1 > 0): #otherwise Y[i+2] is definitely boundary
                prob *= dir_prob(
                        temp_tri_trans[sp1, y, yp1, yp2]
                        # extras4 + extras5
                        + (eq02 & eq13 & eq24) + (eq12 & eq23 & eq34),
                        temp_bi_trans[sp1, yp1, yp2]
                        #+ extras1 + extras2
                        + (eq02 & eq13) + (eq12 & eq23),
                        K, alpha)
                #print "prob_trans_out2", prob_trans_out2
                #prob *= prob_trans_out2
        else: # bigram transition probability
            extras1 = ym1 == y == yp1
            extras2 = ym1 == y
            sum_y = 0
            for  w in range(temp_bi_trans.shape[2]):
                sum_y += temp_bi_trans[sm1, ym1, w]
            prob *=  dir_prob(temp_bi_trans[sm1, ym1, y],
                        sum_y, K, alpha)
            #prob *= prob_trans_in
            #print "prob_trans_in", prob_trans_in
            sum_y = 0
            for  w in range(temp_bi_trans.shape[2]):
                sum_y += temp_bi_trans[sm1, y, w]
            prob *= dir_prob(temp_bi_trans[sm1, y, yp1] + extras1,
                        sum_y + extras2, K, alpha)
            #prob *= prob_trans_out
            #print "prob_trans_out", prob_trans_out

        return prob

    def get_extras(self, int i, CTYPE_t y,
           CTYPE_t ym2, CTYPE_t ym1, CTYPE_t yp1, CTYPE_t yp2):
        """ Unused: left here for clarity about what extras in prob_k are. """

        cdef int eq01 = ym2 == ym1
        cdef int eq12 = ym1 == y
        cdef int eq23 = y == yp1
        cdef int eq34 = yp1 == yp2
        cdef int eq02 = y == ym2
        cdef int eq24 = y == yp2
        cdef int eq13 = ym1 == yp1

        cdef CTYPE_t extras0 = eq01 & eq12
        cdef CTYPE_t extras3 = eq01 & eq12 & eq23
        cdef CTYPE_t extras1 = eq02 & eq13
        cdef CTYPE_t extras4 = eq02 & eq13 & eq24
        cdef CTYPE_t extras2 = eq12 & eq23
        cdef CTYPE_t extras5 = eq12 & eq23 & eq34

        return (extras0, extras1, extras2, extras3, extras4, extras5)

    def log_prob_old(self):
        """ Unused: Returns the log posterior probability of the current state
        of the model. Must match log_prob() below (which is much faster). """
        cdef DTYPE_t trans_probs = 0
        cdef DTYPE_t em_probs = 0

        cdef np.ndarray[CTYPE_t, ndim=2] temp_emissions = np.zeros((self.K,
            self.V), dtype=CTYPE)
        cdef np.ndarray[CTYPE_t, ndim=3] temp_bi_trans = np.zeros((self.D,
            self.K, self.K), dtype=CTYPE)
        cdef np.ndarray[CTYPE_t, ndim=4] temp_tri_trans = np.zeros((self.D, self.K,
            self.K, self.K), dtype=CTYPE)

        cdef CTYPE_t tag, word, stype
        cdef DTYPE_t em_prob, trans_prob

        for i in range(self.N):
            tag = self.Y[i]
            word = self.X[i]
            stype = self.S[i-1]
            if word > 0: # not boundary
                em_prob = dir_prob(temp_emissions[tag, word],
                        sum(temp_emissions[tag, :]), self.V, self.beta[tag])
                em_probs += log(em_prob)
                temp_emissions[tag, word] += 1
            trans_prob = 1
            if self.trigram and i > 1:
               if self.X[i-1] == 0 and stype == 0: # boundary transition
                   trans_prob = 1
               else:
                   trans_prob = dir_prob(temp_tri_trans[stype, self.Y[i-2],
                                                            self.Y[i-1], tag],
                                    sum(temp_tri_trans[stype, self.Y[i-2],
                                        self.Y[i-1], :]), self.K, self.alpha)
            if not self.trigram and i > 0:
                trans_prob = dir_prob(temp_bi_trans[stype, self.Y[i-1], tag],
                    sum(temp_bi_trans[stype, self.Y[i-1], :]), self.K,
                    self.alpha)
            trans_probs += log(trans_prob)

            if self.trigram:
                temp_tri_trans[stype, self.Y[i-2], self.Y[i-1], tag] += 1
            temp_bi_trans[stype, self.Y[i-1], tag] += 1

        # Check that this matches self.log_prob()
        ems_gamma, tr_gamma = self.log_prob()
        print "LP cts: %f, %f" % (em_probs, trans_probs)
        print "LP gms: %f, %f" % (ems_gamma, tr_gamma)
        return (em_probs, trans_probs)

    def log_prob(self):
        """Calculate log prob of the model using the gamma function.
        (In C++ this was slower; in python/cython it's much faster.)
        Note that LP does not include boundary emissions or transitions
        between boundary states (which are probability 1 anyway)."""
        cdef DTYPE_t lp_ems = 0
        cdef DTYPE_t lp_trans = 0
        cdef DTYPE_t beta_y
        cdef CTYPE_t sum_y, e
        cdef DTYPE_t alpha = self.alpha

        # Emission component
        for y in range(self.beta.shape[0]):
            if y > 0: # don't count boundary emissions
                beta_y = self.beta[y]
                lp_ems += self.log_prob_ems_beta_gamma(y, beta_y)

        # Transition component
        lp_trans = self.log_prob_trans_alpha_gamma(alpha)
        return (lp_ems, lp_trans)

    def log_prob_ems_beta_gamma(self, CTYPE_t y, DTYPE_t beta_y):
        """ P(X_y | y, beta_y) = probability of words in y given beta_y. """

        cdef DTYPE_t lp_ems = 0
        cdef CTYPE_t sum_y = 0
        cdef CTYPE_t V = self.V
        cdef np.ndarray[CTYPE_t, ndim=2] temp_emissions = self.emissions
        cdef CTYPE_t x, e
        for x in range(temp_emissions.shape[1]):
            e = temp_emissions[y, x]
            lp_ems += lgamma(e + beta_y) - lgamma(beta_y)
            sum_y += e

        lp_ems += lgamma(beta_y * V) -  lgamma(sum_y + (beta_y * V))

        return lp_ems


    def log_prob_trans_alpha_gamma(self, DTYPE_t alpha):
        """ P(Y | alpha ) = probability of tag sequence given alpha. """
        cdef DTYPE_t lp_trans = 0
        cdef CTYPE_t D = self.D
        cdef CTYPE_t K = self.K
        cdef CTYPE_t sum_y, c
        cdef Py_ssize_t s, t1, t2, t

        cdef np.ndarray[CTYPE_t, ndim=4] temp_tri_trans = self.tri_trans
        cdef np.ndarray[CTYPE_t, ndim=3] temp_bi_trans = self.bi_trans

        # Dir mult dists: each (S, T, T)-> T distribution
        if self.trigram:
            for s in range(temp_tri_trans.shape[0]):
                for t1 in range(temp_tri_trans.shape[1]):
                    for t2 in range(temp_tri_trans.shape[2]):
                        if not (s == 0 and t2 == 0): # not boundary transition
                            sum_y = 0
                            for t in range(temp_tri_trans.shape[3]):
                                c = temp_tri_trans[s, t1, t2, t]
                                lp_trans += lgamma(c + alpha) - lgamma(alpha)
                                sum_y += c
                            lp_trans += lgamma(alpha * K) -  lgamma(sum_y + (alpha * K))
        else:
            for s in range(D):
                for t1 in range(K):
                    if not (s == 0 and t1 == 0): # not boundary transition
                        sum_y = 0
                        for t in range(K):
                            c = temp_bi_trans[s, t1, t]
                            lp_trans += lgamma(c + alpha) - lgamma(alpha)
                            sum_y += c
                        lp_trans += lgamma(alpha * K) -  lgamma(sum_y + (alpha * K))
        return lp_trans

    def sample_hyperparameters(self):
        """Slice-sample HMM/Dir hypers alpha and beta. This samples solely
        from the likelihood, which is equivalent to assuming an "improper
        uniform distribution", a la Goldwater and Griffiths, 2006 (Original
        BHMM paper). (They used a Metropolis Hastings sampler though.) """

        # This stepsize works for Eve corpus (not much stepping out). Tiny
        # test corpora get stuck with overly flat posteriors/likelihoods.
        stepsize = .2

        alpha_samples = slice_sampler.log_slice_sampler(
                self.log_prob_trans_alpha_gamma,
                self.alpha,
                stepsize, self.rng,
                num_samples = 10,
                x_min=VERY_SMALL)
        self.alpha = alpha_samples[self.rng.rng_random_uniform_int(len(alpha_samples))]

        for i in range(1,self.K): # do not resample boundary beta!
            beta_samples = slice_sampler.log_slice_sampler(
                    lambda x: self.log_prob_ems_beta_gamma(i, x),
                    self.beta[i],
                stepsize, self.rng,
                num_samples = 10,
                x_min= VERY_SMALL)
            self.beta[i] = beta_samples[self.rng.rng_random_uniform_int(len(beta_samples))]

    def decrement(self, i, count=-1):
        self.increment(i,count)

    def increment(self, int i, int count=1):
        """ Increments emission, transition matricies based on y[i]. """
        cdef CTYPE_t sm1 = self.S[i-1]
        cdef CTYPE_t si  = self.S[i]
        cdef CTYPE_t sp1 = self.S[i+1]
        cdef CTYPE_t ym2 = self.Y[i-2]
        cdef CTYPE_t ym1 = self.Y[i-1]
        cdef CTYPE_t yi  = self.Y[i]
        cdef CTYPE_t yp1 = self.Y[i+1]
        cdef CTYPE_t yp2 = self.Y[i+2]
        cdef CTYPE_t xi  = self.X[i]
        cdef CTYPE_t N   = self.N

        # this does increment self.tri_trans properly and is faster.
        cdef np.ndarray[CTYPE_t, ndim=4] temp_tri_trans = self.tri_trans
        cdef np.ndarray[CTYPE_t, ndim=3] temp_bi_trans = self.bi_trans
        cdef np.ndarray[CTYPE_t, ndim=2] temp_ems = self.emissions
        cdef np.ndarray[CTYPE_t, ndim=1] temp_ysums = self.y_sums

        temp_ems[yi, xi] += count
        temp_ysums[yi] += count

        # Need to watch out for indicies wrapping around the back.
        if 0 < i:
            temp_bi_trans[sm1, ym1, yi] += count
        if i < N-1:
            temp_bi_trans[si, yi, yp1] += count
        if self.trigram:
            if 1 < i:
                temp_tri_trans[sm1, ym2, ym1, yi] += count
                if CHECK_COUNTS: assert self.tri_trans[self.S[i-1], self.Y[i-2],self.Y[i-1],\
                    self.Y[i]] >= 0, self.tri_trans
            if 0 < i < N-1:
                temp_tri_trans[si, ym1, yi, yp1] += count
                if CHECK_COUNTS: assert self.tri_trans[self.S[i], self.Y[i-1],self.Y[i],\
                    self.Y[i+1]] >=0 , self.tri_trans
            if i < N-2:
                temp_tri_trans[sp1, yi, yp1, yp2] += count
                if CHECK_COUNTS: assert self.tri_trans[self.S[i+1], self.Y[i],self.Y[i+1],\
                    self.Y[i+2]] >= 0, "%d at %d %d %d %d, i=%d; %d %d %d %d" % (
                          self.tri_trans[self.S[i+1], self.Y[i],self.Y[i+1],
                          self.Y[i+2]], self.S[i+1], self.Y[i], self.Y[i+1],
                          self.Y[i+2], i, sp1, yi, yp1, yp2)

    def check_counts(self):
        """ Checks model state, counts.
        More tests are possible (e.g bigram transitions must match
        trigram(s,t,t,:) sums). """
        assert (np.alltrue(self.emissions >= 0))
        assert (np.alltrue(self.bi_trans >= 0))
        assert (np.sum(self.emissions) == self.N)
        assert (np.sum(self.y_sums) == self.N)
        # because the first transitions *in* aren't counted.
        assert (np.sum(self.bi_trans) == self.N-1)

        for y in range(self.K):
            assert np.sum(self.emissions[y,:]) == self.y_sums[y]
        if self.trigram:
            assert (np.alltrue(self.tri_trans >= 0))
            assert (np.sum(self.tri_trans) == self.N-2)

    def print_state(self, output_name=None):
        if output_name:
            outf = open(output_name+".state" , "w")
        else:
            outf = sys.stdout
        outf.write("N=%d K=%d V=%d D=%d\n" % (self.N, self.K, self.V, self.D))
        outf.write("X=%s\n"% str(self.X))
        outf.write("Y=%s\n"% str(self.Y))
        outf.write("alpha=%f beta=%s\n" % (self.alpha, str(self.beta)))
        outf.write("Emmissions\n%s\n" % str(self.emissions))
        outf.write("Bi-transitions\n%s\n" % str(self.bi_trans))
        if self.trigram:
            outf.write("Tri-transitions\n%s\n" % str(self.tri_trans))

    def output_state(self, output_name, word_ids, truetag_ids, stype_ids):
        """ Writes files with output state (final inferred tag sequence).
        output_name.tags: STYPE INFERRED_TAG_ID word TRUE_TAG formatted file
        """
        self.evaluate_vm(output_name)
        self.print_state(output_name)
        outf = open(output_name+".tags" , "w")
        rev_stype_ids = {}
        for s in stype_ids.keys():
            rev_stype_ids[stype_ids[s]] = s
        rev_tag_ids = {}
        for s in truetag_ids.keys():
            rev_tag_ids[truetag_ids[s]] = s
        rev_word_ids = {}
        for w in word_ids.keys():
            rev_word_ids[word_ids[w]] = w

        for i in range(self.N):
            if self.truetags:
                outf.write("%s\t%s\t%s\t%s\n" % (rev_stype_ids[self.S[i]],
                        self.Y[i], rev_word_ids[self.X[i]],
                        rev_tag_ids[self.truetags[i]]))
            else:
                outf.write("%s\t%s\t%s\n" % (rev_stype_ids[self.S[i]],
                        self.Y[i], rev_word_ids[self.X[i]]))

    def evaluate_vm(self, output_name=None):
        assert self.truetags is not None, "Must have true tags to evaluate!"

        # Remove boundary tag at index 0 (hence -1)
        cdef np.ndarray[CTYPE_t, ndim=2] cont_table = np.zeros((self.K-1, self.K-1), dtype=CTYPE)
        for i in range(self.N):
            if self.X[i] > 0:
                cont_table[self.truetags[i]-1, self.Y[i]-1] += 1

        vms = v_measures(cont_table)
        if output_name:
            outf = open(output_name+".vm" , "w")
            outf.write("VM: %.4f VH: %.4f VC: %.4f\n" % (vms[0], vms[1], vms[2]))
            outf.write("\n%s" % str(cont_table))

        return vms[0]

cdef inline DTYPE_t dir_prob(int topcount, int bottomcount, int N, DTYPE_t alpha):
    return (topcount + alpha)/(bottomcount + (N*alpha))

def v_measures(np.ndarray[CTYPE_t, ndim=2] m):
    """Returns v measure elements vm, vc, vh of a contingency matrix m."""
    np.seterr(all='ignore') # avoid annoying warnings

    cdef CTYPE_t sum_m = m.sum()
    cdef np.ndarray[CTYPE_t, ndim=1] gold_counts = m.sum(1)
    cdef DTYPE_t h_c = entropy(gold_counts)/log(2) # entropy of classes (gold)
    cdef np.ndarray[CTYPE_t, ndim=1] found_counts = m.sum(0)
    cdef DTYPE_t h_k = entropy(found_counts)/log(2) # entropy of clusters (estimated)

    cdef DTYPE_t h_c_k = 0 # H(C|K)
    cdef DTYPE_t h_k_c = 0 # H(K|C)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] > 0:
                p_ij = m[i, j]/sum_m
                h_c_k += p_ij * math.log(m[i, j]/found_counts[j], 2)
                h_k_c += p_ij * math.log(m[i, j]/gold_counts[i], 2)

    cdef DTYPE_t vh = 1 if (h_c == 0) else 1 - (-h_c_k/h_c)
    cdef DTYPE_t vc = 1 if (h_k == 0) else 1 - (-h_k_c/h_k)
    cdef DTYPE_t vm = (2 * vh * vc) / (vh + vc)
    return (vm, vh, vc)

def run_sampler(input_file, initialisation, iterations, anneal, randomseed,
        trigram, alpha, beta, samplehyperparameters, output_name):
    """ Function to run sampler: this is called from run_bhmmi.py. """

    words, stypes, truetags, word_ids, tag_ids, stype_ids = read_input(input_file)
    K = len(set(truetags))  # K is number of true tags in input
    V = len(set(words))     # V and K include sentence break tag (from input)
    bhmmi = BHMMI(words, stypes, K, V, alpha, beta, randomseed)

    bhmmi.initialise(init=initialisation, trigram=trigram, truetags=truetags)

    bhmmi.sample(num_iters=iterations, anneal=anneal,
            samplehyperparameters=samplehyperparameters)

    bhmmi.output_state(output_name, word_ids, tag_ids, stype_ids)

def read_input(filename):
    inputlines = []
    word_ids = {}
    stype_ids = {}
    truetag_ids = {}
    firstline = True
    for line in open(filename):
        stype_str, truetag_str, word_str = line.split()
        # First lines must be sentence-breaks (XXX) so they have lowest ids.
        if firstline:
            assert stype_str == "XXX"
            assert truetag_str == "XXX"
            assert word_str == "xxx"
            firstline = False
        stype = stype_ids.setdefault(stype_str, len(stype_ids))
        truetag = truetag_ids.setdefault(truetag_str, len(truetag_ids))
        word = word_ids.setdefault(word_str, len(word_ids))
        inputlines.append([stype,truetag, word])
    stypes, truetags, words= zip(*inputlines)
    return words, stypes, truetags, word_ids, truetag_ids, stype_ids

def main():
    parser = argparse.ArgumentParser(description="BHMM-I Gibbs sampler")
    parser.add_argument("--randomseed", "-R", type=int, default=None)
    parser.add_argument("--anneal", "-A", action="store_true")
    parser.add_argument("--initialisation", "-I", type=str, default="random")
    parser.add_argument("--iterations", "-i", type=int, default=100)
    parser.add_argument("--trigram", "-3", action="store_true")
    parser.add_argument("--alpha", "-a", type=float, default=0.001)
    parser.add_argument("--beta", "-b", type=float, default=0.001)
    parser.add_argument("--samplehyperparameters", "-H", action="store_true")
    parser.add_argument("--output", "-o", type=str, default="out.bhmmi")
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()

    run_sampler(args.input_file, args.initialisation, args.iterations,
            args.anneal, args.randomseed, args.trigram,
            args.alpha, args.beta, args.samplehyperparameters,
            args.output)

if __name__ == "__main__":
    main()

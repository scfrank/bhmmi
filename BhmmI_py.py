#!/usr/bin/python

from __future__ import division

import argparse
import numpy as np
import scipy
import math
import random
import sys

from scipy.stats.distributions import entropy
#import cluster_metrics_cython as cluster_metrics


"""
Note that number of inferred tags (K) is set to be the number of true tags in
the dataset.

Missing ability to do semi-supervised/tag dictionary based learning.

Conditional S-type is *always* i-1 (even in trigrams)


TODO:
    cython prob_k for speed.
    hyperparam re-estimation (is this necessary? - v1 can depend on grid
                    search maybe..?) -> this helps a *lot* though...
    input in treebank format (would be nice to have)

"""

class BHMMI:

    def __init__(self, words, stypes, K, V, alpha, beta):
        """
            words: vector of words/observations, including btw-sentence
                markers (XXX/xxx).
            stypes: vector of sentence types, matched to words.
            K: number of latent states
            V: size of vocabulary
            alpha: dirichlet parameter for transitions
            beta: dirichlet parameter for emissions
        """
        assert (len(words) == len(stypes))
        # make sure the first lines have id 0 for boundary
        assert words[0] == 0
        assert stypes[0] == 0

        self.N = len(words) # Number of word tokens
        self.X = np.array(words, dtype=int)  # Array of words (ids)
        self.Y = np.zeros(self.N, dtype=int) # Array of tags (ids)
        self.S = np.array(stypes, dtype=int) # Array of stypes, co-indexed
                                             # with words and tags
        self.D = len(set(stypes)) # Number of types of sentences
        self.K = K  # Number of types of tags
        # V must be at least as large as input observations
        assert (V >= len(set(self.X)))
        self.V = V # Sie of vocabulary (number of types of words)
        self.alpha = alpha #transition hyperparameter
        self.beta = beta # emission hyperparameter

    def initialise(self, init="random", trigram=False, truetags=False):
        """
            initialisation types are "random", "true", "single" (1 category)
        """
        self.trigram = trigram # boolean whether to use trigram model
        # tag x word matrix of emission probabilities
        self.emissions = np.zeros((self.K, self.V), dtype=int)
        # stype x tag x tag matrix of transition probabilities
        self.bi_trans = np.zeros((self.D, self.K, self.K), dtype=int)
        if self.trigram:
            # stype x tag x tag x tag matrix of transition probabilities
            self.tri_trans = np.zeros((self.D, self.K, self.K, self.K), dtype=int)
        if init == "true" or init == "gold":
            assert truetags, "Missing input true tags for gold initialisation!"
        self.truetags = truetags

        for i in range(self.N):
            if self.X[i] > 0: # not boundary
                if init == "true" or init == "gold":
                    new_y = truetags[i]

                if init == "random":
                    new_y = np.random.randint(1,self.K)
                if init == "single":
                    new_y = 1
                self.Y[i] = new_y

            #print i, self.Y[i], self.X[i], self.K
            # can't use increment because it increments surrounding
            # transitions as well
            #self.increment(i)
            self.emissions[self.Y[i], self.X[i]] += 1
            if 0 < i:
                self.bi_trans[self.bi_at(i)] += 1
            if 1 < i and self.trigram:
                self.tri_trans[self.tri_at(i)] +=1
        self.check_counts()

    def sample(self, num_iters=10, anneal=False):
        #self.print_state()
        print "SAMPLING"

        if anneal:
            if num_iters < 20:
                print "Not enough iterations to anneal: not annealling, T=1"
                anneal = False
            else:
                # Copying annealling schedule from Sharon's code.
                annealling_schedule = [1.0] + [1/(1.03**i) for i in range(19)]
                annealling_schedule.reverse()

        temp = 1
        for iteration in range(num_iters):
            if (iteration % 10) == 0:
                em_probs, trans_probs = self.log_prob()
                print "iter: ", iteration, "LP: %.2f Em: %.2f Trans: %.2f" % (em_probs
                        + trans_probs, em_probs, trans_probs)
                print "VM:", self.evaluate_vm()
            if anneal and iteration % (num_iters//20) == 0:
                if iteration < (num_iters//20)*20:
                    temp = annealling_schedule[iteration // (num_iters//20)]
                    print "iter %d: Changing temperature to %f" % (iteration, temp)
            for i in range(self.N):
                if self.X[i] > 0: # not boundary (don't resample boundaries)
                    self.decrement(i)
                    new_y = self.sample_y(i, temp)
                    self.Y[i] = new_y
                    self.increment(i)
                    #print i, self.X[i], self.Y[i]
                    #self.print_state()
                    self.check_counts()
        print "FINISHED"
        em_probs, trans_probs = self.log_prob()
        print "Final sample: LP: %.2f Em: %.2f Trans: %.2f" % (em_probs
                        + trans_probs, em_probs, trans_probs)
        print "Final VM:", self.evaluate_vm()
        self.check_counts()
        #self.print_state()

    def sample_y(self, i, temp=1):
        """ TODO: Remember to worry about underflow -> need to sample in log space?"""
        probs = np.zeros(self.K, float)
        for y in range(1, self.K):
            probs[y] = self.prob_k(i, y)
            probs[y] = math.pow(probs[y], temp)
            assert probs[y] > 0, "Zero prob/underflow? at i: %d, tag y %d,\
                                  prob %f" % (i, y, probs[y])
        # renormalise probs
        probs = [p/np.sum(probs) for p in probs]
        new_y = np.argmax(np.random.multinomial(1, probs))
        #print i, probs, new_y
        return new_y

    def prob_k(self, i, y):
        x = self.X[i]
        prob = 1
        # P(x | y)
        prob_em = dir_prob(self.emissions[y, x],
                np.sum(self.emissions[y,:]), self.V, self.beta)
        #print "prob_em", prob_em
        prob *= prob_em
        if self.trigram:
            # extra transition counts
            eq01 = self.Y[i-2] == self.Y[i-1]
            eq12 = self.Y[i-1] == y
            eq23 = y == self.Y[i+1]
            eq34 = self.Y[i+1] == self.Y[i+2]
            eq02 = y == self.Y[i-2]
            eq24 = y == self.Y[i+2]
            eq13 = self.Y[i-1]  == self.Y[i+1]
            extras = np.zeros(6, int)
            extras[0] = eq01 & eq12
            extras[3] = eq01 & eq12 & eq23
            extras[1] = eq02 & eq13
            extras[4] = eq02 & eq13 & eq24
            extras[2] = eq12 & eq23
            extras[5] = eq12 & eq23 & eq34
            prob_trans_in = dir_prob(
                    self.tri_trans[self.tri_at(i, y, y_at=2)],
                    self.bi_trans[self.bi_at(i, y, y_at=1)],
                self.K, self.alpha)
            #print "prob_trans_in", prob_trans_in
            prob *= prob_trans_in
            prob_trans_out1 = dir_prob(
                self.tri_trans[self.tri_at(i+1, y, y_at=1)] +
                extras[3],
                self.bi_trans[self.bi_at(i+1,y, y_at=0)]
                + extras[0],
                self.K, self.alpha)
            #print "prob_trans_out1", prob_trans_out1
            prob *= prob_trans_out1
            if (self.Y[i+1] > 0): #otherwise Y[i+2] is definitely boundary
                prob_trans_out2 = dir_prob(
                    self.tri_trans[self.tri_at(i+2, y, y_at=0)] +
                    extras[4] + extras[5],
                    self.bi_trans[self.bi_at(i+2)] +
                    extras[1] + extras[2],
                    self.K, self.alpha)
                #print "prob_trans_out2", prob_trans_out2
                prob *= prob_trans_out2
        else: # bigram transition probability
            extra1 = self.Y[i-1] == y == self.Y[i+1]
            extra2 = self.Y[i-1] == y
            prob_trans_in = dir_prob(self.bi_trans[self.bi_at(i, y, y_at=1)],
                    sum(self.bi_trans[self.S[i-1], self.Y[i-1], :]), self.K,
                        self.alpha)
            prob *= prob_trans_in
            #print "prob_trans_in", prob_trans_in
            prob_trans_out = dir_prob(self.bi_trans[self.bi_at(i+1, y,
                y_at=0)] + extra1,
                sum(self.bi_trans[self.S[i], y, :]) + extra2,
                self.K, self.alpha)
            prob *= prob_trans_out
            #print "prob_trans_out", prob_trans_out

        return prob

    def log_prob(self):
        """ Returns the log posterior probability of the current state of the model. """
        trans_probs = 0
        em_probs = 0

        temp_emissions = np.zeros((self.K, self.V), dtype=int)
        temp_bi_trans = np.zeros((self.D, self.K, self.K), dtype=int)
        if self.trigram:
            temp_tri_trans = np.zeros((self.D, self.K, self.K, self.K), dtype=int)

        for i in range(self.N):
            tag = self.Y[i]
            word = self.X[i]
            stype = self.S[i-1]
            if word > 0: # not boundary
                em_prob = dir_prob(temp_emissions[tag, word],
                        sum(temp_emissions[tag, :]), self.V, self.beta)
                em_probs += math.log(em_prob)
                temp_emissions[tag, word] += 1
            trans_prob = 1
            if self.trigram and i > 1:
                if self.X[i-1] == 0 and word == 0: # boundary transition
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
            trans_probs += math.log(trans_prob)


            if self.trigram:
                temp_tri_trans[stype, self.Y[i-2], self.Y[i-1], tag] += 1
            temp_bi_trans[stype, self.Y[i-1], tag] += 1

        return (em_probs, trans_probs)



    def decrement(self, i, count=-1):
        self.increment(i,count)

    def increment(self, i, count=1):
        """ Is there a cleaner way of doing this? Must ensure i-1 etc doesn't
        end up with negative indices that wrap from the back! """
        #print "trans coords:", i, count, self.S[i], self.Y[i-2], self.Y[i-1],\
        #        self.Y[i], self.Y[i+1], self.Y[i+2]

        self.emissions[self.Y[i], self.X[i]] += count
        if 0 < i:
            self.bi_trans[self.S[i-1], self.Y[i-1], self.Y[i]] += count
        if i < self.N-1:
            self.bi_trans[self.S[i], self.Y[i], self.Y[i+1]] += count
        if self.trigram:
            if 1 < i:
                self.tri_trans[self.S[i-1], self.Y[i-2],self.Y[i-1], self.Y[i]] += count
                assert self.tri_trans[self.S[i-1], self.Y[i-2],self.Y[i-1],\
                    self.Y[i]] >= 0, self.tri_trans
            if 0 < i < self.N-1:
                self.tri_trans[self.S[i], self.Y[i-1],self.Y[i], self.Y[i+1]] += count
                assert self.tri_trans[self.S[i], self.Y[i-1],self.Y[i],\
                    self.Y[i+1]] >=0 , self.tri_trans
            if i < self.N-2:
                self.tri_trans[self.S[i+1], self.Y[i],self.Y[i+1], self.Y[i+2]] += count
                assert self.tri_trans[self.S[i+1], self.Y[i],self.Y[i+1],\
                    self.Y[i+2]] >= 0, self.tri_trans

    def bi_at(self, i, y=-1, y_at=1):
        """
        Returns bigram (stype, tag/state, tag/state)
        corresponding to s[i-1], y[i-1], y[i];
            if y is -1, use current y values, else y=argument;
            y_at is 0,1, corresponding to which y value should be replaced.
            i.e. if y_at=1 (and y= isn't -1), use y argument at last item in
            bigram; if y_at=0, replace y at the first item in bigram.
        """
        assert y_at in [0,1], "Bad y_at argument (not 0,1): %d" % y_at
        assert y in range(-1,self.K), "Bad y argument (not -1...%d): %d" %\
                (self.K-1, y)
        bigram = [self.Y[i-1], self.Y[i]]
        stype = self.S[i-1]
        if y >=0:
            bigram[y_at] = y
        return tuple([stype] + bigram)

    def tri_at(self, i, y=-1, y_at=2):
        """
        Returns trigram (stype, tag/state, tag/state, tag/state)
        corresponding to s[i-1], y[i-2], y[i-1], y[i];
            if y is -1, use current y values, else y=argument;
            y_at is 0,1,2, corresponding to which y value should be replaced.
            i.e. if y_at=2 (and y= isn't -1), use y argument at last item in
            trigram; if y_at=0, replace y at the first item in trigram.
        """
        assert y_at in [0,1,2], "Bad y_at argument (not 0,1,2): %d" % y_at
        assert y in range(-1,self.K), "Bad y argument (not -1...%d): %d" %\
                (self.K-1, y)
        trigram = [self.Y[i-2], self.Y[i-1], self.Y[i]]
        stype = self.S[i-1]
        if y >=0:
            trigram[y_at] = y
        return tuple([stype] + trigram)


    def check_counts(self):
        assert (np.alltrue(self.emissions >= 0))
        assert (np.alltrue(self.bi_trans >= 0))
        assert (np.sum(self.emissions) == self.N)
        # because the first transitions *in* aren't counted.
        assert (np.sum(self.bi_trans) == self.N-1)
        if self.trigram:
            assert (np.alltrue(self.tri_trans >= 0))
            assert (np.sum(self.tri_trans) == self.N-2)

        #   assert (np.sum(self.tri_trans[self.S[i-1], :, self.Y[i-1], y]) ==
        #       self.bi_trans[self.S[i-1], self.Y[i-1], y],
        #       (np.sum(self.tri_trans[self.S[i-1], :, self.Y[i-1], y]),
        #       self.bi_trans[self.S[i-1], self.Y[i-1], y],
        #       ))

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
        """ Writes files with output state.
        output_name.tags: STYPE INFERRED_TAG_ID word TRUE_TAG formatted file
        TODO: other files?
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

        #Make a matrix where m(i,j) = count(i=gold tag,j=inferred tag).
        # XXX assuming K is same for gold tags and inferred tags!
        # -1 to remove boundary tags (at index 0)
        # float is because cluster_metrics is stupid here. (should be int0
        cont_table = np.zeros((self.K-1, self.K-1), dtype=float)
        for i in range(self.N):
            if self.X[i] > 0:
                cont_table[self.truetags[i]-1, self.Y[i]-1] += 1

        #vms = cluster_metrics.v_measures(cont_table)
        vms = v_measures(cont_table)
        if output_name:
            outf = open(output_name+".vm" , "w")
            outf.write("VM: %.4f VH: %.4f VC: %.4f\n" % (vms[0], vms[1], vms[2]))
            outf.write("\n%s" % str(cont_table))

        return vms[0]


def dir_prob(topcount, bottomcount, N, alpha):
    return (topcount + alpha)/(bottomcount + (N*alpha))

def v_measures(m):
    """A lot slower than cython'd version; here so the code is selfsufficient
    for now."""
    np.seterr(all='ignore') # avoid annoying warnings
    """Returns v measure elements vm, vc, vh of a contingency matrix m."""

    c_size, k_size = m.shape
    sum_m = float(m.sum())
    gold_counts = m.sum(1)
    h_c = entropy(gold_counts)/math.log(2) # entropy of classes (gold)
    found_counts = m.sum(0)
    h_k = entropy(found_counts)/math.log(2) # entropy of clusters (estimated)

    h_c_k = 0 # H(C|K)
    h_k_c = 0 # H(K|C)
    for i in range(c_size):
        for j in range(k_size):
            if m[i, j] > 0:
                p_ij = m[i, j]/sum_m
                h_c_k += p_ij * math.log(m[i, j]/found_counts[j], 2)
                h_k_c += p_ij * math.log(m[i, j]/gold_counts[i], 2)

    vh = 1 if (h_c == 0) else 1 - (-h_c_k/h_c)
    vc = 1 if (h_k == 0) else 1 - (-h_k_c/h_k)
    vm = (2 * vh * vc) / (vh + vc)
    return (vm, vh, vc)

def run_sampler(input_file, initialisation, iterations, anneal, trigram, alpha, beta,
        output_name):
    print "RUNNING PYTHON VERSION"

    words, stypes, truetags, word_ids, tag_ids, stype_ids = read_input(input_file)
    K = len(set(truetags)) # K is number of true tags
    V = len(set(words)) # V and K include sentence break tag
    bhmmi = BHMMI(words, stypes, K, V, alpha, beta)
    bhmmi.initialise(init=initialisation, trigram=trigram, truetags=truetags)

    bhmmi.sample(num_iters=iterations, anneal=anneal)

    bhmmi.output_state(output_name, word_ids, tag_ids, stype_ids)

def read_input(filename):
    inputlines = []
    word_ids = {}
    stype_ids = {}
    truetag_ids = {}
    firstline = True
    for line in open(filename):
        # stypes, tags, words strings are changed to integer ids here for
        # faster lookup
        #inputlines.append(line.split())
        stype_str, truetag_str, word_str = line.split()
        # Make sure first lines are sentence-breaks (XXX) so they have lowest
        # ids.
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
    parser.add_argument("--output", "-o", type=str, default="out.bhmmi")
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()

    if args.randomseed:
        random.seed(args.randomseed)

    run_sampler(args.input_file, args.initialisation, args.iterations,
            args.anneal, args.trigram, args.alpha, args.beta, args.output)

if __name__ == "__main__":
    main()

#!/usr/bin/python

# for profiling
import pstats, cProfile

import BhmmI

if __name__ == "__main__":

    profiling = False

    input_file = "testfile"

    initialisation = "random"
    iterations = 50
    anneal = False
    trigram = True
    alpha = 0.1
    beta = 0.1
    output = "out.bhmmi"
    samplehyperparameters = True
    randomseed = None

    if not profiling:
        # skips BhmmI.main() and argparse arguments
        BhmmI.run_sampler(input_file, initialisation, iterations,
            anneal, randomseed, trigram, alpha, beta, samplehyperparameters, output)
    else:
        randomseed = 26
        s = "BhmmI.run_sampler(input_file, initialisation, iterations, anneal, randomseed, trigram, alpha, beta, samplehyperparameters, output)"
        cProfile.run(s, "bhmmi.prof")

        s = pstats.Stats("bhmmi.prof")
        s.strip_dirs().sort_stats("time").print_stats(20) # (10):top 10 functions only


This is a python/cython rewrite of the BHMM-I from:

Adding sentence types to a model of syntactic category
acquisition. Stella Frank, Sharon Goldwater, Frank Keller. TopiCS in Cognitive
Science 5 (3), pp. 495--521. 2013.

If you use this code, please cite the above paper.

#Cython version (BhmmI.pyx):

Cython files (`*.pyx`) need to be compiled. To do this, run:

```
python setup_slicesampler.py build_ext --inplace
python setup_hasrng.py build_ext --inplace
python setup_BhmmI.py build_ext --inplace
```

The model is then run from the `run_bhmmi.py` script. This script
specifies the parameters input to the BHMM-I model, like number of
iterations and hyperparameter values. It should be self explanatory;
the arguments match the python arguments below, plus a
samplehyperparameters (boolean) argument.


#Pure python version - BhmmI_py.py:

Much much slower! Also does not do hyperparameter estimation.

To run, do `python BhmmI_py.py -h` in a commandline (you'll then see the
commandline arguments):

```
BHMM-I Gibbs sampler

positional arguments:
  input_file (see below for notes on input_file format)

optional arguments:
  -h, --help            show this help message and exit
  --randomseed RANDOMSEED, -R RANDOMSEED  this is for debugging
  --anneal, -A  This makes inference find a better solution faster
  --initialisation INITIALISATION, -I INITIALISATION
        Initialisation options are:
          "gold" to start from gold tags (for debugging mostly)
          "single" to start with all tokens in same category
          "random" all tokens start in randomly assigned categories
  --iterations ITERATIONS, -i ITERATIONS Number of iterations to run
                                         the sampler
  --trigram, -3   Use trigram model (default is bigram)
  --alpha ALPHA, -a ALPHA   Value for alpha (transition) hyperparameter
  --beta BETA, -b BETA      Value for beta (emission) hyperparameter
  --output OUTPUT, -o OUTPUT  Filename to write output to
```

#Input/output format:
  The input should be a file formatted like this (see testfile for an example):

```
  XXX XXX xxx
  XXX XXX xxx
  D   DT  THIS
  D   V   is
  D   DT  an
  D   JJ  example
  D   NN  sentence
  XXX XXX xxx
  XXX XXX xxx
```

Each line consists of three columns: the first column encodes the
current sentence type (e.g D, Q, but it doesn't matter what strings
you use); the second encodes the true part of speech tags, and the
third has the word tokens (lowercased and otherwise tokenised).
Between each utterance there are two lines of boundary markers (the
`XXX`s).

The output format  of the `X.tag` is nearly the same, except that each
line has four columns:

`STYPE INFERRED_TAG_ID word TRUE_TAG`

The first and third columns should match in input and output. The
second column contains the cluster ids found by the sampler, and the
fourth column should match the true tags in the second column of the
input.

##Output files
- `[out].tag` contains the tags as inferred in the last iteration, as described
above.
- `[out].state` contains final hyperparameter values and counts (model state).
- `[out].vm` contains the final VM score, as well as a confusion matrix of
  inferred tags vs gold tags.


#Code details:

HMM structure: $N$ observed values $X$, latent values $Y$
  - Each $x_i$ has an associated $y_i$ and observed sentence type $s_i$
  - (Sentence type is constant throughout a sentence)

Conditional probability of latent state:
  - P(y_i = y) = P(x_i | y) P(y | y_(i-1), y_(i-2), s_i) [trigram]*
  -  ... plus dependencies on previous + following transitions!

BHMM-I Gibbs sampler consists of:

Initialisation: random, gold, single value

Loop for $j$ iterations:
  - Calculate log probablity of current model
  - Loop over $n$ datapoints:
      - Decrement current $y_i$ value from emission and transition
    distribution
      - Calculate P(y_i = y) for each possible state y:
        P(y_i = y) = P(x_i | y) P(y | y_(i-1), y_(i-2), s_i) [trigram]*
        ... plus dependencies on previous + following transitions!
      - Sample new $y_i$ from this probability distribution
      - Increment transitions, emissions, using new $y_i$

After sampling:
  - evaluate - VM only right now.

Objects:
  - X, Y, S vectors
  - emission: K x V matrix (K states, V words)
  - transitions: K x K (bigram), KxKxK (trigram)


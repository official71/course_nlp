import math
import nltk
import time
from collections import defaultdict

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# Return {item : count} in given list
# Because the list.count() method within for loop is too slow
def count_list_items(lst):
    dd = defaultdict(float)
    for i in lst:
        dd[i] += 1

    return dd

# Return unigram, bigram or trigram tuple of given sentence
# The number of grams is specified by argument n
def sentence_to_ngram(str, n):
    l = [START_SYMBOL, START_SYMBOL]
    l.extend(str.rstrip().split(' '))
    l.extend([STOP_SYMBOL])

    r = []
    if n == 1:
        r = l[2:]
    elif n == 2:
        r = list(nltk.bigrams(l[1:]))
    elif n == 3:
        r = list(nltk.trigrams(l))

    return r

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    # ngram tuples
    unigram_tuples = []
    bigram_tuples = []
    trigram_tuples = []
    num_start_symbols = 0
    for tc in training_corpus:
        unigram_tuples.extend(sentence_to_ngram(tc, 1))
        bigram_tuples.extend(sentence_to_ngram(tc, 2))
        trigram_tuples.extend(sentence_to_ngram(tc, 3))
        num_start_symbols += 1

    # ngram counts
    unigram_counts = count_list_items(unigram_tuples)
    bigram_counts = count_list_items(bigram_tuples)
    trigram_counts = count_list_items(trigram_tuples)
    # Manually assign the count of START_SYMBOL in unigram and bigram for calculation of 
    # bigram and trigram probabilities
    unigram_counts[START_SYMBOL] = float(num_start_symbols)
    bigram_counts[(START_SYMBOL, START_SYMBOL)] = float(num_start_symbols)

    # ngram log probabilities
    unigram_total = len(unigram_tuples)
    for u in unigram_counts:
        unigram_p[u] = math.log(unigram_counts[u]/unigram_total, 2)

    for b in bigram_counts:
        n = unigram_counts[b[0]]
        bigram_p[b] = math.log(bigram_counts[b]/n, 2) if n else MINUS_INFINITY_SENTENCE_LOG_PROB

    for t in trigram_counts:
        n = bigram_counts[t[0:2]]
        trigram_p[t] = math.log(trigram_counts[t]/n, 2) if n else MINUS_INFINITY_SENTENCE_LOG_PROB

    # print "\nA1 verifications"
    # print unigram_p['captain']
    # print unigram_p['captain\'s']
    # print unigram_p['captaincy']
    # print bigram_p[('and', 'religion')]
    # print bigram_p[('and', 'religious')]
    # print bigram_p[('and', 'religiously')]
    # print trigram_p[('and', 'not', 'a')]
    # print trigram_p[('and', 'not', 'by')]
    # print trigram_p[('and', 'not', 'come')]

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []

    for c in corpus:
        s = 0
        for g in sentence_to_ngram(c, n):
            p = ngram_p.get(g, 'NA')
            if p == 'NA':
                s = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            s += p
        scores.append(s)

    # print "\nA2 verifications " + str(n)
    # print scores[0]
    # print scores[1]
    # print scores[2]

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

#TODO: IMPLEMENT THIS FUNCTION    
# Calculcates the perplexity of a language model
# scores_file: one of the A2 output files of scores 
# sentences_file: the file of sentences that were scores (in this case: data/Brown_train.txt) 
# This function returns a float, perplexity, the total perplexity of the corpus
def calc_perplexity(scores_file, sentences_file):
    # score data
    infile = open(scores_file, 'r')
    scores = infile.readlines()
    infile.close()

    log_prob = 0
    count = 0
    i = 0
    # traverse sentences
    infile = open(sentences_file, 'r')
    for str in infile:
        count += len(str.split(' '))
        log_prob += float(scores[i])
        i += 1
    infile.close()

    # 2 ** ((-1/<word counts>) * <sum of all log probabilities>)
    perplexity = 2 ** (-log_prob / count)
    return perplexity 

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    ulambda = blambda = tlambda = 1
    factor = ulambda + blambda + tlambda

    for c in corpus:
        s = 0
        unigram_tuples = sentence_to_ngram(c, 1)
        bigram_tuples = sentence_to_ngram(c, 2)
        trigram_tuples = sentence_to_ngram(c, 3)

        for i in range(len(unigram_tuples)):
            uni_log_prob = unigrams.get(unigram_tuples[i], 'NA')
            bi_log_prob = bigrams.get(bigram_tuples[i], 'NA')
            tri_log_prob = trigrams.get(trigram_tuples[i], 'NA')

            # encountering a new unigram, set whole sentence to -inft
            if uni_log_prob == bi_log_prob == tri_log_prob == 'NA':
                s = MINUS_INFINITY_SENTENCE_LOG_PROB
                break

            # apply lambda factors to probabilities
            uni_prob = ulambda * (2 ** uni_log_prob) if uni_log_prob != 'NA' else 0
            bi_prob = blambda * (2 ** bi_log_prob) if bi_log_prob != 'NA' else 0
            tri_prob = tlambda * (2 ** tri_log_prob) if tri_log_prob != 'NA' else 0
            p = float(uni_prob + bi_prob + tri_prob) / factor
            s += math.log(p, 2)

        scores.append(s)

    # print "\nA3 verifications"
    # print scores[0]
    # print scores[1]
    # print scores[2]
    # print scores[3]
    # print scores[4]

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()

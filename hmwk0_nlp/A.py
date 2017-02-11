import nltk
import sys

greeting = sys.stdin.read();
print greeting

token_list = nltk.word_tokenize(greeting)
squirrel = 0
girl = 0
print "The tokens in the greeting are"
for token in token_list:
    print token
    tlower = token.lower()
    if tlower == 'squirrel':
        squirrel += 1
    elif tlower == 'girl':
        girl += 1

print "There were %d instances of the word 'squirrel' and %d instances of the word 'girl.'" % (squirrel, girl)

import sys
from solutionsA import calc_perplexity

def main():
    if len(sys.argv) < 2:
        print "Usage: python perplexity.py <file of scores> <file of sentences that were scored>"
        exit(1)
    perplexity = calc_perplexity(sys.argv[1], sys.argv[2]) 

    print "The perplexity is", perplexity    
	 
if __name__ == "__main__": main()

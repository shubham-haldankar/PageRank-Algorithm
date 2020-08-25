import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pd= dict()
    
    for key in corpus:
        pd[key]=round((1-damping_factor)/len(corpus),4)

    if len(corpus[page])==0:
        for key in corpus:
            pd[key]+=damping_factor/len(corpus)
    else:
        for key in corpus[page]:
            pd[key]+=damping_factor/len(corpus[page])
            
    return pd
    #raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    surfer= random.choices(list(corpus.keys()))[0]
    pagerank= dict()
    for key in corpus:
        pagerank[key]= 0
        
    for i in range(0,n):
        pagerank[surfer]+= 1 
        pd= transition_model(corpus, surfer, damping_factor)
        surfer= random.choices(list(pd.keys()),weights=pd.values(),k=1)[0]
    
    for key, value in pagerank.items():
        pagerank[key]= round(value/n,4)
    
    return pagerank
    #raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N= len(corpus)
    pagerank= dict()
    for key in corpus:
        pagerank[key]= round(1/N,4)
    
    iterate= 1
    while iterate:
        new_pagerank= dict()
        iterate= 0
        for key in corpus:
            old= pagerank[key]
            
            new_pagerank[key]= round((1-damping_factor)/N,4)
            for i in corpus:
                if key in corpus[i]:
                    new_pagerank[key]+= round(damping_factor*pagerank[i]/len(corpus[i]),4)
            
            if abs(new_pagerank[key]-old)>0.001:
                iterate= 1
        for i in corpus:
            pagerank[i]= new_pagerank[i]
    return pagerank
    #raise NotImplementedError


if __name__ == "__main__":
    main()

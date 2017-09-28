# SimonData Home Test Assignment


## Question:
The homework assignment involves writing some code against Etsy’s API, parsing results, and then deriving some aggregate stats from the data:

1. Sign up for an account on Etsy and obtain a set of API keys.
2. Spend a few minutes browsing Etsy, and identify a set of 10 different shops on the site.
3. Use the API to pull all items sold in the shop and extract the title and descriptions from these items. 
4. With the above dataset, write an algorithm to identify the top 5 meaningful terms for each shop. 
5. Write a program to display the results (CLI or Web UI).



## Solution:

I used the TF-IDF algorithm to determine the most important (meaningful) words of each store.
To calculate the scores of each word, I both implemented the algorithm, and used the scikit-learn library's implementation.
Both give similar results, but not surprisingly, scikit's implementation is 2 orders of magnitude faster.
 
 
 
My Solution includes both a command line python program, and a jupyter notebook [here](https://github.com/lfcunha/etsy_meaningful_words/blob/master/notebook.ipynb)
 
 
## Usage
 
 1) This program requires >  python3.4
 2) install requirements
 ```bash
    > pip3 install -r requirements.txt
```
 3) running the program:
 ```bash
> python3 run.py [--tfidf scikit|diy [--level 10|20|30|40]]
```
 The arguments are optional. Default tfidf to run is the scikit implementation. Default log level is INFO
 
 
 4) run tests:
 ```bash
    > pytest
```
 
 5) If you have any problem running the program locally, you can:
    1. run the docker container with the provided DockerFile
        ```bash
        > docker build . etsy-image
        > docker run -it --rm --name etsy --entrypoint=/bin/bash etsy-image
        ```
        At this point you will be inside the docker container.
        ```bash
        > cd /opt/etsy
        > python run.py [--tfidf scikit|diy [--level 10|20|30|40]]
        ```
        
    2. reachout to me lfcunha @ gmail.com
    3. and/or visualize the output in the jupyter notebook



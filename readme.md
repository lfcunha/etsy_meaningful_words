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
 
 
 
My Solution includes both a command line python program, and a jupyer notebook [here]()
 
 
 
 


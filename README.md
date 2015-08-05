# Bipartite Link Prediction
This project explores various methods for bipartite link prediction using Yelp's Dataset Challenge dataset. In particular, we try to predict which restaurants a particular user will review. This was done as a class project for Stanford's Social and Information Network Analysis class ([CS224W](http://web.stanford.edu/class/cs224w/index.html)). The final report is included in the repository (writeups/final_report.pdf). It requires scikit-learn, networkx, and snap.py to run.

## Running
1. Place Yelp academic datasets in data/provided
2. Run dataset_maker.py to generate examples
3. Run any of the following files:
  * dataset_metrics.py (prints various properties of the dataset)
  * random_baseline.py (random predictions)
  * random_walks.py (make predictions using unsupervised random walks)
  * similarity.py (make predictions using heuristic similarity measures)
  * supervised_classifier.py (make predictions using a supervised binary classifier)
  * supervised_random_walks.py (make predictions using supervised random walks, see [Backstrom and Leskovec, 2011](http://arxiv.org/abs/1011.4071))
  * svd.py (make predictions using matrix factorization)
4. Use eval.py to generate model evaluation metrics.

# AppReviewAnalysis
**Analyze App Reviews**

## Topic Clustering Parameter Search
This project is designed to perform a parameter search for topic clustering algorithms. The main function takes a list of documents and returns the optimal parameters for clustering the documents based on their topics.
### Features

- Utilizes popular topic modeling algorithms such as Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF).
- Performs a grid search to find the optimal parameters for the chosen topic modeling algorithm.
- Evaluates the performance of the algorithm using coherence scores.
- Returns the optimal parameters for the given dataset.

### Dependencies

To run this script, you will need the following Python libraries:

- `numpy`
- `pandas`
- `gensim`
- `sklearn`


### Output

The function returns a dictionary containing the optimal parameters for the topic clustering algorithm. The dictionary includes the following keys:

- `algorithm`: The optimal topic modeling algorithm (e.g., 'LDA' or 'NMF').
- `num_topics`: The optimal number of topics for the given dataset.
- `alpha`: The optimal alpha value for the chosen algorithm (only applicable for LDA).
- `beta`: The optimal beta value for the chosen algorithm (only applicable for LDA).

### Example

```python
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning is a type of machine learning.",
    "Neural networks are used in deep learning.",
    "Python is a popular programming language for machine learning.",
    "TensorFlow and PyTorch are popular deep learning frameworks."
]

optimal_parameters = inTopic(documents)

print(optimal_parameters)
```

Output:

```python
{
    'algorithm': 'LDA',
    'num_topics': 2,
    'alpha': 0.1,
    'beta': 0.01
}
```

This output indicates that the optimal topic modeling algorithm for the given dataset is LDA with 2 topics, and the optimal alpha and beta values are 0.1 and 0.01, respectively.

* **Disclaimers**:
  - Unauthorized use of the code in this repository is not permitted. If any laws are violated as a result, the user will be solely responsible for the consequences.
  - Due to confidentiality agreements and commercial secrecy, sensitive algorithms or information/data have been removed or replaced in this project. What remains are the algorithmic processes developed using open-source tools. If the Python script files or the project appears incomplete, it's due to the removal of algorithmic content that may potentially lead to legal disputes. The README files corresponding to each code may retain more implementation methods or details. Your understanding is greatly appreciated.

# CS4248 Project: Labelled Unreliable News (LUN)

AY22/23 Semester 2

Based on data from: https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection

Paper: https://aclanthology.org/2021.acl-long.62/

Common Crawl Word Embeddings Stanford gloVe (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip

Our repository is organised as follows:

- `neural_networks/` contains the jupyter notebooks for building our neural network models using Tensorflow
- `models.ipynb` contains our main models using scikit-learn
- `data_exploration.ipynb` contains some preliminary data exploration and analysis
- `custom_feature_models.ipynb` contains the custom features we created
- `word_embeddings.ipynb` contains our implementations using word2vec an gloVe embeddings
- `requirements.txt` contains the required Python packages to run our code

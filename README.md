# Comparative Topic Modelling Based on Long COVID Tweets

#### This study compares different topic modeling techniques — LDA, NMF, LSA, Top2Vec, and BERTopic — applied to tweets related to Long COVID. The goal is to explore and evaluate how these models capture topics within social media discourse on this important health issue.
```
├── .gitignore
├── .python-version
├── README.md
├── requirements.txt
├── notebooks/
│ ├── 00_setup.ipynb
│ ├── 01_load_and_merge_data.ipynb
│ ├── 02_general_preprocessing.ipynb
│ ├── 03_minimal_preprocessing.ipynb
│ ├── 04_lda.ipynb
│ ├── 05_nmf.ipynb
│ ├── 06_lsa.ipynb
│ ├── 07_top2vec.ipynb
│ ├── 08_reduced_top2vec.ipynb
│ ├── 09_BERTopic.ipynb
│ ├── 10_reduced_BERTopic.ipynb
├── src/
│ └── utils/
│ ├── data_loader.py
│ ├── topic_diversity.py
```
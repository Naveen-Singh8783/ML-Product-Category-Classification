#ML-Product-Category-Classification

This project focuses on predicting Etsy product categories using text features from product titles, descriptions, and tags.
The goal was to build models that can classify both top-level and bottom-level categories based on product metadata.

Overview

Etsy has thousands of product listings organized in a large category hierarchy.
Using NLP techniques, this project builds two models:

Top Category Model – predicts 15 broad product categories

Bottom Category Model – predicts over 2,600 detailed categories

Both models use a simple yet effective TF-IDF + Logistic Regression pipeline.

Method

Data Cleaning:

Combined title, description, and tags into one text column

Lowercased text and removed missing values

Feature Extraction:

Used TF-IDF with unigrams and bigrams (15,000 features max)

Modeling:

Logistic Regression with balanced class weights

Different solvers for top and bottom levels for efficiency

Evaluation:

Used macro and weighted F1 scores for fair performance comparison

Results
Task	F1 Score
Top Category	0.86
Bottom Category	0.59

The top-level model performed very well, while the bottom-level model struggled with class imbalance — common for large category hierarchies.

Next Steps

Try transformer models (e.g. BERT, RoBERTa)

Handle class imbalance with data augmentation

Add structured fields (like material or occasion)

Explore hierarchical classification approaches

🧩 Files

├── ML_model.ipynb            

├── ML_assignment_report.pdf    

└── README.md                      

Tech Stack

Python

scikit-learn

pandas, numpy

matplotlib, seaborn

Author

Naveen Singh
ML-Labs, School of Computing, Dublin City University

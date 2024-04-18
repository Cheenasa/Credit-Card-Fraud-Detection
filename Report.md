Credit Card Fraud Detection.

Introduction:

 This assignment explores various machine learning concepts such as weak learners, boosting, bagging, oversampling, under sampling, and dataset splitting to minimize data leakage. Additionally, the use of performance metrics to evaluate model effectiveness was explored. The dataset chosen for this task was the Credit Card Fraud dataset obtained from Kaggle, consisting of 568,630 rows and 31 columns.

Key Concepts Covered
1.	Weak Learner: A weak learner is a model that performs slightly better than random guessing on a classification or regression task. Weak learners are often used as building blocks in ensemble learning methods, such as boosting and bagging.
2.	Bagging: Bagging, short for Bootstrap Aggregating, is an ensemble learning technique that aims to improve the stability and accuracy of machine learning algorithms by reducing variance and overfitting.
3.	Boosting: Boosting is an ensemble learning technique that aims to improve the performance of machine learning models by combining the predictions of multiple weak learners sequentially.

Data Preprocessing and Resampling:
 The first step in the analysis was to ensure the cleanliness of the dataset by checking for duplicate entries and null values. The dataset was split into three categories based on transaction amount: transactions with amounts less than $1000, transactions with amounts ranging from $1000 to $5000, and transactions with amounts greater than $5000. To address the class imbalance present in the dataset, we employed resampling techniques. Specifically, oversampling  was applied to the dataset containing transactions with amounts ranging from $1000 to $5000, and under sampling to the dataset containing transactions with amounts greater than $5000.

Evaluation:
Model performance evaluation relied on confusion matrices and key performance metrics, including accuracy, precision, recall, and F1-score. However, more attention was paid to the recall metric, as predicting that a transaction is not fraudulent when it is, could enable criminal activities and result in heavy losses to the bank. Therefore, the focus was on increasing recall or minimizing false negatives to identify true positive (fraudulent) transactions.

Model Building and Findings:
 The machine learning models were built using various algorithms, including decision trees, bagging, boosting, and random forests. For each algorithm, I constructed models using the original dataset as well as the resampled datasets, resulting in a total of five datasets for evaluation.

Key findings:

•	The models performed well, with high recall, accuracy, F1-score, precision, and ROC score (above 95%).

•	Under sampling emerged as the most effective technique for generating robust models across the datasets.

•	Random forest consistently outperformed other algorithms across all datasets.

Conclusion and Recommendations:
Based on the analysis, the following conclusions and recommendations can be made:
1.	Under sampling emerged as the most effective resampling technique for generating robust models across the datasets.
2.	The random forest model trained on the under sampled dataset achieved the best performance, with high recall, accuracy, F1-score, precision, and ROC score.
3.	It is recommended to deploy the random forest model trained on the under sampled dataset for production use, as it provides the most reliable and accurate credit card fraud detection capabilities.




# Biomarker Imputation using Transformer vs KNN

### Motivation and Goal

Having done limited work in bioinformatics, I wanted to explore this intersection from data and AI perspective. Therefore, the goal of this project is to explore the problem of imputation for missing clinical blood biomarker values. To do this, I utilise synthetic dataset inspired and crafted from the MIMIC-IV ICU dataset. The project compares traditional and deep learning-based approaches while reconstructing missing biomarker values. With the goal of evaluating weather transformer-based approaches can outperform classical methods like K-Nearest Neighbour (KNN) in accuracy and generalisation.

### Dataset creation and exploration

Firstly, in order to create a synthetic dataset, I filtered and sorted different lab events for unique patients across different tables, I extracted 10 most common blood biomarker data found within critical ICU patients. Focusing on data related to blood fluid and blood gas category for simplicity reasons.

#### Here are the most common blood biomarkers
![](https://github.com/Ronnn007/Biomarker-Imputation-Transformer-vs-KNN/blob/main/Graphs/Most%20common%20biomarkers.png)


#### Synthetic Dataset

For transformer-based approaches to produce comparable results, I generated 5000 records based on the filtered MIMIC-IV dataset. 
This filtered data included columns such as:

The biomarker value, Volume unit, Healthy references for lower and higher ranges, Biomarker label, patient Age and Gender.

Using this realistic ICU laboratory data and its normal/log-normal distributions, I generated the synthetic data. I also sampled Age and Gender distributions based on realistic patient demographics. Additionally, here is an overview of the generated data:

|   |  Min	| Max| Mean|
|---|---|---|---|
| Age  |  21.00  | 91.00 | 60.73|
|Cortisol | 2.0  |28.08| 9.978096|
| Creatine Kinase (CK)|20.00|	7590.69|	324.767116|
| Ferritin  |10.00|	3642.36|	239.720760|
|  Free Calcium |0.85|	1.40|	1.196686|
|  Glucose |50.00	|198.98|	100.919200|
| Haemoglobin | 5.00|	18.00|	11.915046|
| Lipase  | 0.63|	400.00|	43.600038|
|  Monocytes |0.00	|21.39|	8.003792|
|  Red Blood Cells |	3.15|	6.50|	5.010670|

### Summary of Methods implemented

|  Method |  Description | Loss  | Data Input Type|
|---|---|---|---|
|KNN imputer (Traditional)|	Utilising feature similarity and Euclidean distance|	N/A|	Continuous values|
|Transformer (Discrete Tokenization)|	Data discretised into bins using BERT style masked token prediction|	Cross-Entropy|	Token ID, using custom tokenizer|
|Transformer (Continuous Regression)|	Direct numeric regression on masked input values|	Mean Squared Error|	Continuous values|


### Results and comparison

The model performance was evaluated using Mean Absolute error and Root Mean square Error on masked test values. Additionally results were also normalized using standard deviation to account for scale different between different biomarkers. For instance Normalized_MAE = (mae / std)

#### Here is a results heatmap output of all mothods

![](https://github.com/Ronnn007/Biomarker-Imputation-Transformer-vs-KNN/blob/main/Graphs/Results%20heatmap.png)

### Discussion
Overall, the Discrete transformer performs poorly as the tokenization and binning resulted in a loss of granularity. Sharp boundaries likely caused this, separating values near the edges into different classes. Therefore, the MAE performance for this model measures extreme results for biomarkers such as Free Calcium and Red Blood Cells. These features have very narrow ranges compared to others. Additionally, some bins appeared less frequently, producing sparse embedding and weak contextual learning.

Regression-based KNN and Transformer models, on the other hand, performed better by predicting real-valued outputs and preserving the natural ordering and continuity of the data. The continuous Transformer in particular leveraged its attention mechanism to learn cross-feature dependencies, capturing nonlinear physiological relationships between biomarkers.

Finally, KNN performs competitively on low-variance biomarkers because of its reliance on local neighbourhood similarity. However, it struggles to capture global nonlinear dependencies that the transformer can learn through self-attention.
This experiment showcases that continuous deep learning approaches generalises better for multi-biomarker imputation tasks, particularly when underlying variables are interdependent have complex relationships.

### Future directions

Future directions could involve utilising hybrid models such as transformer and auto-encoder, as well as imputation on complete MIMIC-IV dataset, which would involve more complex biomarker relationships. 

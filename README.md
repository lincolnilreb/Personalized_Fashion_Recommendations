# Personalized Fashion Recommendations

The midpoint notebook contains the preprocessing techniques and clustering methods implemented.

Implementation is present at

`Data_Cleaning.ipynb`: Contains the preprocessing techniques and clustering methods implemented.

`Kmeans.ipynb`: Contains the Kmeans clustering and recommendation system implementation.

`LGBM_KNN.ipynb`: Contains the LGBM and KNN recommendation system implementation.

`Metrics.ipynb`: Contains the evaluation metrics for the recommendation system.

Here’s a list of the libraries used in the notebook along with the installation instructions.

### Libraries
1. **Data Analysis and preprocessing**
   - `pandas`
   - `numpy`
   - `sklearn`
2. **Dimensionality Reduction**
   - `sklearn.decomposition` (for `PCA`)
   - `sklearn.manifold` (for `TSNE`)
3. **Clustering**
   - `sklearn.cluster` (for `KMeans`, `DBSCAN`)
4.**Classification**
   - `sklearn.neighbors` (for `KNN`)
   - `lightgbm.sklearn` (for `LGBM`)
5. **Evaluation Metrics**
   - `sklearn.metrics` (for `silhouette_score`)
6. **Visualization**
   - `matplotlib`
   - `seaborn`
   - `plotly` (for interactive 3D plots)

### Installation Instructions

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly wordcloud lightgbm
```

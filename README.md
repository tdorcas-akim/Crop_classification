Crop vs. Non-Crop Classification from Satellite Time Series
What the project is about

This project builds and compares machine learning models that decide whether a place on the ground is farmland, using one year of satellite, weather, and terrain data. The data come from CropHarvest, an open dataset that brings together crop labels from many sources and pairs them with satellite data. We compare classic machine learning with deep learning on this binary classification task. The models are trained on places from around the world, and they are then tested on Kenya and Togo, which are kept out of training, to see how well they work on African places they have never seen. After cleaning, the dataset has 81,813 places, and about 70% of them are farmland.

What the notebook does

1. Downloads the CropHarvest data(or uses the zipped copy from Google Drive)
2. Checks and cleans the data: duplicates, conflicting labels, extreme values
3. Scales the data using the training set only
4. Keeps Kenya and Togo(4,582 places) aside as a test set on Africa
5. Splits the rest into training, validation, and test sets (70/15/15, seed 42)
6. Builds 128 summary features per place for the classic models
7. Trains classic models: logistic regression, Random Forest and gradient boosting(scikit-learn)
8. Trains neural networks: MLP, 1D CNN, GRU(TensorFlow and Keras, with tf.data pipelines)
9. Runs 17 experiments, compared on the same validation set
10. Ends with the final evaluation, learning curves, confusion matrices, ROC curves and an error analysis by data source
    
Main results

Measure	          Gradient boosting (E8)   	Wide 1D CNN (D5)
Validation AUC	        0.931	                  0.932
Test AUC	              0.933	                  0.935
Test accuracy	          0.869	                  0.870
Test F1	                0.910	                  0.910
Kenya AUC	              0.873	                  0.865
Togo AUC	              0.833                   0.865

The two best models are almost tied on the test set.
Both lower on Kenya and Togo than on the test set.
About 62% of the test errors come from one crowdsourced source (Geo-Wiki).
The full tables and figures are in the notebook and in the report.

What is in the repository

Crop_Classification_Sum_Project.ipynb: the notebook
requirements.txt: the Python packages and their versions
README.md: this file
.gitignore: keeps data and model files out of the repository

How to run it

Open the notebook in Google Colab.
Choose a GPU: Runtime, Change runtime type, T4 GPU.
Use Runtime, Run all. 
The first cells mount your Google Drive and create a folder called crop_project, where everything is saved. cropharvest_raw.zip is already in that folder so the notebook uses it.

To install the packages yourself:

bash
pip install -r requirements.txt
pip install cropharvest --no-deps

cropharvest is installed with --no-deps because it asks for an old version of pandas (below 2.0), which does not work with the Python version of the current Colab.

Links

Google Drive folder (dataset, best models, logs): https://drive.google.com/drive/folders/18YBMjHq-SudKS4I6S0JHSRmbJ7YDAExV?usp=drive_link
Video presentation: 

Credit

Data: G. Tseng, I. Zvonkov, C. Nakalembe, and H. Kerner, "CropHarvest: A global dataset for crop-type classification," NeurIPS Datasets and Benchmarks Track, 2021 (Creative Commons Attribution-ShareAlike 4.0 license)
Main libraries: scikit-learn, TensorFlow and Keras, pandas, NumPy, geopandas
The full list of references is in the written report

# covid-econometrics

## Data download

download_data.ipynb: download cac40 and covid19 csv files.

## Data ingestion

import_csv_to_hdfs.ipynb: import a csv to hdfs.

nifi_template_ingest_files_into_hdfs_elasticsearch.xml: Nifi template to import local file into hdfs and index them into elasticsearch automatically.

## Data exploration, cleaning and analysis

cac40_data_analysis.ipynb: import cac40 csv, clean data and plot cac40 price over covid pandemic period.

covid_data_analysis.ipynb: import covid csv, clean data and plot number of covid cases during the pandemic.

join_cac40_covid_data_for_ML_analysis.ipynb: join cac40 and covid clean data over the same time points, data preprocessing for Machine learning analysis.

## Machine learning analysis

Linear regression analysis to study the relation between CAC40 price and the number of covid cases.
Use cases with different libraries:

### Scikitlearn - python

cac40_covid_linear_regression_scikit_learn.ipynb

### Pyspark - Spark python

cac40_covid_linear_regression_pyspark_MLlib.ipynb

### SparkML - Spark scala

cac40_covid_linear_regression_sparkML_1_feature.scala
cac40_covid_linear_regression_sparkML_4_features.scala


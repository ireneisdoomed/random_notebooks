import gcsfs
import joblib
import pandas as pd
import shap

CHROM_TO_CLASSIFIER = {
    '1': 'xgboost-full_model-high_medium-0.model.joblib.gz',
    '2': 'xgboost-full_model-high_medium-3.model.joblib.gz',
    '3': 'xgboost-full_model-high_medium-0.model.joblib.gz',
    '4': 'xgboost-full_model-high_medium-2.model.joblib.gz',
    '5': 'xgboost-full_model-high_medium-4.model.joblib.gz',
    '6': 'xgboost-full_model-high_medium-1.model.joblib.gz',
    '7': 'xgboost-full_model-high_medium-3.model.joblib.gz',
    '8': 'xgboost-full_model-high_medium-0.model.joblib.gz',
    '9': 'xgboost-full_model-high_medium-2.model.joblib.gz',
    '10': 'xgboost-full_model-high_medium-1.model.joblib.gz',
    '11': 'xgboost-full_model-high_medium-2.model.joblib.gz',
    '12': 'xgboost-full_model-high_medium-3.model.joblib.gz',
    '13': 'xgboost-full_model-high_medium-4.model.joblib.gz',
    '14': 'xgboost-full_model-high_medium-4.model.joblib.gz',
    '15': 'xgboost-full_model-high_medium-4.model.joblib.gz',
    '16': 'xgboost-full_model-high_medium-1.model.joblib.gz',
    '17': 'xgboost-full_model-high_medium-3.model.joblib.gz',
    '18': 'xgboost-full_model-high_medium-2.model.joblib.gz',
    '19': 'xgboost-full_model-high_medium-4.model.joblib.gz',
    '20': 'xgboost-full_model-high_medium-1.model.joblib.gz',
    '21': 'xgboost-full_model-high_medium-2.model.joblib.gz',
    '22': 'xgboost-full_model-high_medium-2.model.joblib.gz',
    'X': 'xgboost-full_model-high_medium-4.model.joblib.gz',
}
MODEL_BUCKET_NAME = 'gs://genetics-portal-dev-staging/l2g/220128/models'
FM_BUCKET_NAME = (
    'gs://genetics-portal-dev-staging/l2g/220712/gold_standards/featurematrix_w_goldstandards.full.220712.parquet'
)


def main(variant: str):

    # Load model
    model_file_name = get_model_file_name(variant)
    model = load_model(MODEL_BUCKET_NAME, model_file_name)
    chroms_fold_test = model['run_info']['fold_test_chroms']

    # Load feature matrix to calculate shap values
    fm = load_feature_matrix(FM_BUCKET_NAME, chroms_fold_test)

    # Compute SHAP values
    shap_values = compute_shap_values(model, fm)



    pass



def get_model_file_name(variant: str):
    chrom = variant.split('_')[0]
    return CHROM_TO_CLASSIFIER[chrom]


def load_model(bucket_name, file_name):
    """
    It opens a file in the specified bucket and loads it into a joblib object
    
    :param bucket_name: The name of the bucket where the model is stored
    :param file_name: the name of the file you want to load
    :return: The model is being returned.
    """
    with fs.open(f'{bucket_name}/{file_name}') as f:
        return joblib.load(f)


def load_feature_matrix(bucket_path: str, chroms_fold):
    """
    It loads the feature matrix from the bucket, filters it to only include the chromosomes in the fold,
    and returns the first 50 rows
    
    :param bucket_path: the path to the bucket where the feature matrix is stored
    :type bucket_path: str
    :param chroms_fold: a list of chromosomes to use for training and testing
    :return: A dataframe with the first 50 rows of the data.
    """

    parquet_files = ['gs://' + x for x in fs.ls(bucket_path) if x.endswith('.parquet')]
    fm = pd.concat([pd.read_parquet(x).query('chrom in @chroms_fold') for x in parquet_files], ignore_index=True).head(50)
    return fm

def compute_shap_values(model, fm):
    """
    It takes a model and a feature matrix, and returns the SHAP values for the feature matrix
    
    :param model: the model you want to explain
    :param fm: the feature matrix
    :return: The shap_values are the contribution of each feature to the prediction.
    """
    explainer = shap.TreeExplainer(model['model'].best_estimator_)
    shap_values = explainer.shap_values(fm)
    return shap_values

if __name__ == '__main__':

    global fs
    fs = gcsfs.GCSFileSystem()

import pickle

import joblib
import gcsfs
import pandas as pd
from PIL import Image
import shap
import streamlit as st
from streamlit_shap import st_shap

L2G_VERSION_TO_EXPLAINERS = {
    '220712': '2022-07-29',
    '220803': '2022-08-11',
}

def get_explainer_version(l2g_version:str):
  """
  > Given a version of the L2G model, return the version of the explainer that should be used to
  explain it
  
  Args:
    l2g_version (str): The version of the L2G model you want to use.
  
  Returns:
    The explainer version that corresponds to the l2g version.
  """
  return L2G_VERSION_TO_EXPLAINERS[l2g_version]

def user_input():
    """
    It takes two inputs from the user, a variant ID and a study ID, and returns them as a tuple
    
    Returns:
      A tuple of the variant ID and locus ID of interest.
    """
    variant_id = st.text_input('Write a valid Variant ID in the chrom_pos_ref_alt format:', value='1_154839804_C_T')
    study_id = st.text_input('Write a valid study ID:', value='FINNGEN_R6_I9_AF')
    if variant_id and study_id:
        return variant_id, study_id


def load_shap_values(bucket_name, file_name):
    """
    It loads the shap values from a file in a GCS bucket
    SHAP Values are precomputed because this calculation is slow (~3 minutes).
    
    Args:
      bucket_name: The name of the GCS bucket where the model is stored.
      file_name: the name of the file you want to load
    
    Returns:
        A shap._explanation.Explanation object which essentially is a numpy matrix of shape (n_samples, n_features)
    """
    with gcsfs.GCSFileSystem().open(f'{bucket_name}/{file_name}', 'rb') as f:
        return pickle.load(f)


@st.cache(suppress_st_warning=True)
def load_predictions_from_gcs(file_path):
    """
    It loads a table with all the L2G predictions and their features into a pandas dataframe
    
    Args:
      file_path: The path to the file on GCS.
    
    Returns:
      A dataframe
    """
    return pd.read_parquet(file_path)


def find_prediction(feature_matrix: pd.DataFrame, variant_id: str, study_id: str):
    """
    It takes a feature matrix and a variant and returns the row of the feature matrix that corresponds
    to that variant

    Args:
      feature_matrix (pd.DataFrame): the feature matrix that you want to use to make predictions
      variant (str): a string of the form 'chr_pos_ref_alt'
      study_id (str): The study ID of the study you want to predict on.

    Returns:
      A dataframe with the row of the feature matrix that matches the variant and study_id
    """
    chrom, pos, ref, alt = variant_id.split('_')

    prediction = feature_matrix.query(
        'chrom == @chrom & pos == @pos & ref == @ref & alt == @alt & study_id == @study_id'
    )
    if len(prediction) == 0:
        raise ValueError(f'No prediction found for variant {variant_id} and study {study_id}')
    return prediction


def display_summary_plot(prediction: pd.DataFrame):
    """
    It displays the summary plot for the model that handles loci for the following chromosomes: 3, 8, 1
    
    Args:
      prediction (pd.DataFrame): a dataframe with the following columns:
    """
    shap_values_fold = prediction['model_fold'].iloc[0]
    chroms_subset = ', '.join([str(e) for e in prediction['model_fold_chroms'].iloc[0]])

    summary_plot = Image.open(f'plots/summary_plot_fold_{shap_values_fold}.png')

    st.image(
        summary_plot,
        caption=f'Feature importance for the model that handles loci for the following chromosomes: {chroms_subset}.',
    )


def load_model(bucket_name: str, file_name: str):
    """
    It opens a file from a GCS bucket, and loads it into a joblib object
    
    Args:
      bucket_name (str): The name of the GCS bucket where the model is stored.
      file_name (str): The name of the file you want to load.
    
    Returns:
      The model is being returned.
    """
    with gcsfs.GCSFileSystem().open(f'{bucket_name}/{file_name}') as f:
        return joblib.load(f)


def get_model_features(bucket_name: str, file_name: str):
    """
    Given a bucket name and a file name, load the model from the file and return the features used to
    train the model
    
    Args:
      bucket_name (str): The name of the GCS bucket where the model is stored.
      file_name (str): The name of the file you want to load.
    
    Returns:
      A list of features used in the model.
    """
    model = load_model(bucket_name, file_name)
    return model['run_info']['features']

@st.cache(suppress_st_warning=True)
def cache_model_features(predictions: pd.DataFrame, model_bucket: str, model_filename: str):
    """
    This function caches the predictions relevant to the L2G model.
    It first checks that all features are present in the provided feature matrix.

    Args:
      predictions (pd.DataFrame): the dataframe of predictions that you want to score
      model_bucket (str): The name of the GCS bucket where the model is stored.
      model_filename (str): The name of the model file in the model bucket.
    
    Returns:
      The model features are being returned.
    """

    l2g_features = get_model_features(model_bucket, model_filename)
    feature_match =  all(feature in predictions.columns  for feature in l2g_features)
    if not feature_match:
        raise ValueError(f'The model features {l2g_features} are not in the predictions dataframe')
    return l2g_features


def display_force_plot(shap_values: shap._explanation.Explanation, prediction: pd.DataFrame, features: list):
    """
    It takes in a shap explanation object, a prediction dataframe, and a list of features, and then it
    displays a force plot of the shap values for the desired prediction
    
    Args:
      shap_values (shap._explanation.Explanation): the shap values for the model
      prediction (pd.DataFrame): the prediction you want to explain
      features (list): list of features to display in the plot
    """

    shap_value_index = prediction['index'].iloc[0]
    base_value = prediction['explainer_expected_value'].iloc[0]

    st_shap(
        shap.force_plot(
            base_value=base_value,
            shap_values=shap_values[shap_value_index, :].values,
            features=prediction.filter(items=features),
            link='logit',
        )
    )
    st.caption(f"Force plot of the SHAP values for {prediction['gene_id'].iloc[0]}")
    st.write(
        """
        The chart above shows how features contributed to the model's prediction for a specific observation.
        - The output value is the L2G score for that observation
        - The expected value is the mean of all predictions and informs about the value that would be predicted if we did not know any features for the current output.
        - Features that push the prediction higher (to the right) are shown in red, and those pushing the prediction lower are in blue.
        """
    )

def display_bar_plot(shap_values: shap._explanation.Explanation, prediction: pd.DataFrame):

    shap_value_index = prediction['index'].iloc[0]

    st_shap(
        shap.plots.bar(
            shap_values[shap_value_index],
            max_display=100,
            show=False
        )
    )
    st.caption(f"Bar plot of the SHAP values for {prediction['gene_id'].iloc[0]}")
    st.write(
        """
        The chart above shows why an observation receives its prediction given its variable values by plotting the shapley values. Compared with the waterfall plot, the bar plot centers at zero to show the contributions of variables.
        """
    )

def display_waterfall_plot(shap_values: shap._explanation.Explanation, prediction: pd.DataFrame):

    shap_value_index = prediction['index'].iloc[0]

    st_shap(
        shap.plots.waterfall(
            shap_values[shap_value_index],
            max_display=100,
            show=False
        )
    )
    st.caption(f"Waterfall plot of the SHAP values for {prediction['gene_id'].iloc[0]}")
    st.write(
        """
        The chart above shows why an observation receives its prediction given its variable values. You start with the bottom of a waterfall plot and add (red) or subtract (blue) the values to get to the final prediction.
        - The output value at the top is the L2G score for that observation
        - The expected value in the bottom is the mean of all predictions and informs about the value that would be predicted if we did not know any features for the current output.
        - The contribution of each feature to the output value is plotted. Each value represent the deviation of that feature in that observation with respect with its mean.
        """
    )
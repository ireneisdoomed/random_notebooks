import pandas as pd

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

    return feature_matrix.query('chrom == @chrom & pos == @pos & ref == @ref & alt == @alt & study_id == @study_id')

import streamlit as st

from src.utils import *

data_to_l2g_version = {
    '2022-07-29': '220712',
    '2022-08-11': '220803',
}


st.title('Locus-2-Gene Explainability')
st.write('This app allows you to explore the explainability of a L2G prediction using SHAP. ')

# Set constants
DATA_VERSION = '2022-08-11'
BUCKET_PATH = f'gs://ot-team/irene/l2g_explainability/{DATA_VERSION}'
L2G_BUCKET = 'gs://genetics-portal-dev-staging/l2g/220712/models'

with st.sidebar:
    with st.form(key='my_form'):
        st.header('Query L2G:')
        variant_id = st.text_input('Write a valid Variant ID in the chrom_pos_ref_alt format:', value='1_154839804_C_T')
        study_id = st.text_input('Write a valid study ID:', value='FINNGEN_R6_I9_AF')
        submit_button = st.form_submit_button(label='Submit')

if submit_button:

    fm = load_predictions_from_gcs(f'{BUCKET_PATH}/l2g_enriched_fm.parquet')
    # I extract the model features from a random fold of the L2G model, all the folds contain the same ones
    model_features = cache_model_features(
        predictions=fm, model_bucket=L2G_BUCKET, model_filename='xgboost-full_model-high_medium-0.model.joblib.gz'
    )

    # Find the row of the feature matrix that corresponds to the variant and study_id
    prediction = find_prediction(fm, variant_id, study_id)

    # Get the SHAP values for the prediction
    shap_values_name = prediction['shap_values_filename'].iloc[0]
    shap_values = load_shap_values(BUCKET_PATH, shap_values_name)

    st.header('Local feature contribution')
    st.write(
        """
        Features for each loci are not distributed uniformly. For example, the model is biased towards considering distance-related features with higher importance just because they are always present, whereas features that describe variant pathogenicity or colocalising molecular QTLs are less important because they are less present.

        We want to explore the contribution of each feature for each prediction as I expect this to vary depending on the features' presence.
        """
    )
    # Display highest scoring gene
    predicted_gene = prediction['gene_id'].iloc[0]
    url = f'https://genetics.opentargets.org/gene/{predicted_gene}'
    st.write(
        f"The highest scoring gene is: {[predicted_gene]}({url})",
    )

    # Plot the SHAP values
    bar_plot, waterfall_plot, force_plot = st.tabs(['Bar plot', 'Waterfall plot', 'Force plot'])
    with bar_plot:
        st.header('Bar plot')
        display_bar_plot(shap_values, prediction)

    with waterfall_plot:
        st.header('Waterfall plot')
        #display_waterfall_plot(shap_values, prediction)
        st.write('WIP. Currently debugging.')

    with force_plot:
        st.header('Force plot')
        display_force_plot(shap_values, prediction, model_features)

    st.header('Global feature contribution')
    with st.expander('See global contribution of the features to the model'):
        display_summary_plot(prediction)
        st.write(
            """
         The chart above shows the relative importance of the different features in the L2G model based on the mean of each shap value.
        """
        )

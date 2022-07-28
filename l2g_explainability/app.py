import streamlit as st

from src.utils import *

st.title('Locus-2-Gene Explainability')

# Set constants
DATA_VERSION = '2022-07-28'
FM_PATH = f'gs://ot-team/irene/l2g_explainability/{DATA_VERSION}/l2g_enriched_fm.parquet'


variant_id = st.text_input('Write a valid Variant ID in the chrom_pos_ref_alt format:', value='1_154839804_C_T')
study_id = st.text_input('Write a valid study ID:', value='FINNGEN_R6_I9_AF')

# Load ft
fm = pd.read_parquet(FM_PATH)


prediction = find_prediction(fm, variant_id, study_id)

# Global explainability section

# Local explainability section

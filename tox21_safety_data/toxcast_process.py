from typing import Tuple
import pandas as pd
import numpy as np

# Datasets paths
assays_path = "INVITRODB_V3_3_SUMMARY/assay_methods_invitrodb_v3_3.xlsx"
targets_path = "INVITRODB_V3_3_SUMMARY/gene_target_information_invitrodb_v3_3.xlsx"
quality_stats_path = "INVITRODB_V3_3_SUMMARY/Assay_Quality_Summary_Stats_200819.csv" 

# Columns of interest
assays_cols = [
    "aeid", "assay_component_endpoint_name", "acid",
    "assay_component_name", "assay_function_type", "intended_target_family",
    "intended_target_family_sub", "assay_component_desc", "assay_design_type",
    "biological_process_target", "organism", "tissue",
    "cell_format", "cell_short_name", "assay_format_type"]
targets_cols = ["official_symbol", "aenm"]
quality_stats_cols = ["aenm", "acnt"]

def load_data(
    assays_path: str,
    assays_cols: list,
    targets_path: str,
    targets_cols: list,
    quality_stats_path: str,
    quality_stats_cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assays = pd.read_excel(assays_path,
                            sheet_name="assay_merge",
                            engine = 'openpyxl')[assays_cols]

    targets = pd.read_excel(targets_path,
                            sheet_name="Sheet 1",
                            engine = 'openpyxl')[targets_cols]

    quality_stats = pd.read_csv(quality_stats_path)[quality_stats_cols]
    return assays, targets, quality_stats

def enrich_assays(
    assays: pd.DataFrame,
    targets: pd.DataFrame,
    quality_stats: pd.DataFrame
) -> pd.DataFrame:
    assays_enriched = (assays
        # Merge assays with targets and quality_stats table
        .merge(
            targets,
            left_on="assay_component_endpoint_name",
            right_on="aenm",
            how="inner")
        .merge(
            quality_stats,
            left_on="assay_component_endpoint_name",
            right_on="aenm",
            how="inner"
        )
        # Drop duplicates
        .drop_duplicates(
            subset=["acid", "biological_process_target", "official_symbol"]
        )
    )
    return assays_enriched

def filter_assays(
    assays: pd.DataFrame
) -> pd.DataFrame:
    assays = (assays
        # Filter human assays
        .query('organism == "human"')
        # Filter out endpoints describing features associated with assay cytotoxic effects
        .query('intended_target_family_sub != "cytotoxicity" or biological_process_target != ["cell death", "cell proliferation"]')
        # Filter out endpoints describing features associated with assay background effects
        .query('assay_function_type != "background control" or assay_design_type != ["background reporter", "viability reporter"] or intended_target_family != "background measurement"')
        # Filter out assays without active samples
        .query('acnt != 0')
    )
    return assays

def adjust_tissue(
    assays: pd.DataFrame
):
    '''
    The tissue value is dropped whenever the assay is cell based.
    The rationale to test in a cell line is often due to the availability and stability
    of the cell line, so the inference between a cell line and the tissue is incorrect
    '''
    assays.loc[~assays["cell_short_name"].isna(), "tissue"] = np.nan


if __name__ == '__main__':
    assays, targets, quality_stats = load_data(
        assays_path, assays_cols,
        targets_path, targets_cols,
        quality_stats_path, quality_stats_cols
    )
    assays_enriched = enrich_assays(assays, targets, quality_stats)
    adjust_tissue(assays_enriched)
    out = filter_assays(assays_enriched)
    #print(out.shape)
    #print(out["official_symbol"].nunique())
    out.to_csv("ToxCast_filtered.tsv", sep="\t", header=True, index=False)    

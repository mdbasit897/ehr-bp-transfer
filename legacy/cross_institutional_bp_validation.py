#!/usr/bin/env python3

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, QuantileRegressor, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import wasserstein_distance, ks_2samp
import warnings
import logging
from datetime import datetime
import os
import yaml
import json
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================================================
# FEATURE SET DEFINITIONS
# ============================================================================

@dataclass
class FeatureSetConfig:
    """Define different feature sets for ablation studies"""

    # Core physiological features (always available)
    VITAL_SIGNS = [
        'heart_rate_mean', 'heart_rate_std', 'heart_rate_cv',
        'resp_rate_mean', 'resp_rate_std',
        'spo2_mean', 'spo2_std',
        'temperature_mean', 'temperature_std'
    ]

    LABORATORY_FEATURES = [
        'sodium', 'potassium', 'chloride', 'bicarbonate',
        'creatinine', 'bun', 'hemoglobin', 'wbc', 'platelet',
        'glucose', 'lactate'
    ]

    MEDICATION_FEATURES = [
        'vasopressor_count', 'beta_blocker', 'diuretic',
        'ace_inhibitor', 'arb', 'calcium_channel_blocker'
    ]

    TEMPORAL_FEATURES = [
        'measurement_frequency', 'hr_lability_events'
    ]

    DEMOGRAPHIC_FEATURES = [
        'age', 'gender'
    ]

    # Institution-specific features (NOT available in eICU or unreliable)
    INSTITUTIONAL_CATEGORICAL = [
        'first_careunit', 'insurance', 'language', 'ethnicity'
    ]

    # Derived interaction terms (may have missing data issues)
    DERIVED_INTERACTIONS = [
        'age_creatinine_interaction', 'hr_creatinine_interaction'
    ]

    # Features available in BOTH MIMIC and eICU (16 features - THE CRITICAL SET)
    COMMON_FEATURES_16 = [
        # Vitals (8)
        'heart_rate_mean', 'heart_rate_std', 'heart_rate_cv',
        'resp_rate_mean', 'spo2_mean', 'temperature_mean',
        'measurement_frequency', 'hr_lability_events',
        # Labs (6)
        'creatinine', 'bun', 'hemoglobin', 'glucose', 'lactate', 'sodium',
        # Demographics (2)
        'age', 'gender'
    ]

    @classmethod
    def get_feature_set(cls, feature_set_name: str) -> List[str]:
        """Get specific feature set by name"""
        if feature_set_name == 'all_28':
            return (cls.VITAL_SIGNS + cls.LABORATORY_FEATURES +
                    cls.MEDICATION_FEATURES + cls.TEMPORAL_FEATURES +
                    cls.DEMOGRAPHIC_FEATURES + cls.INSTITUTIONAL_CATEGORICAL +
                    cls.DERIVED_INTERACTIONS)
        elif feature_set_name == 'current_22':
            # Excludes institutional categorical (4) and derived interactions (2)
            return (cls.VITAL_SIGNS + cls.LABORATORY_FEATURES +
                    cls.MEDICATION_FEATURES + cls.TEMPORAL_FEATURES +
                    cls.DEMOGRAPHIC_FEATURES)
        elif feature_set_name == 'intersection_16':
            return cls.COMMON_FEATURES_16
        elif feature_set_name == 'no_meds_17':
            # Exclude medications
            return (cls.VITAL_SIGNS + cls.LABORATORY_FEATURES +
                    cls.TEMPORAL_FEATURES + cls.DEMOGRAPHIC_FEATURES)
        elif feature_set_name == 'no_labs_16':
            # Exclude labs
            return (cls.VITAL_SIGNS + cls.MEDICATION_FEATURES +
                    cls.TEMPORAL_FEATURES + cls.DEMOGRAPHIC_FEATURES)
        else:
            raise ValueError(f"Unknown feature set: {feature_set_name}")

    @classmethod
    def get_available_sets(cls) -> List[str]:
        """List all available feature sets"""
        return ['all_28', 'current_22', 'intersection_16', 'no_meds_17', 'no_labs_16']


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

class ExperimentTracker:
    """Track all experiments with metadata and results"""

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.experiments = []
        self.results_file = self.output_dir / 'experiment_results.json'

        # Load existing experiments if available
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.experiments = json.load(f)

    def log_experiment(self, exp_id: str, config: Dict, results: Dict):
        """Log experiment configuration and results"""
        experiment = {
            'exp_id': exp_id,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results
        }
        self.experiments.append(experiment)
        self._save()

    def _save(self):
        """Save experiments to disk"""
        with open(self.results_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """Retrieve specific experiment"""
        for exp in self.experiments:
            if exp['exp_id'] == exp_id:
                return exp
        return None

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table of all experiments"""
        rows = []
        for exp in self.experiments:
            row = {
                'exp_id': exp['exp_id'],
                'feature_set': exp['config'].get('feature_set', 'unknown'),
                'imputation': exp['config'].get('imputation_method', 'unknown'),
                'dataset': exp['config'].get('dataset', 'unknown'),
                'mimic_sbp_r2': exp['results'].get('mimic', {}).get('SBP', {}).get('R2', np.nan),
                'mimic_sbp_rmse': exp['results'].get('mimic', {}).get('SBP', {}).get('RMSE', np.nan),
                'eicu_sbp_r2': exp['results'].get('eicu', {}).get('SBP', {}).get('R2', np.nan),
                'eicu_sbp_rmse': exp['results'].get('eicu', {}).get('SBP', {}).get('RMSE', np.nan),
                'mimic_dbp_r2': exp['results'].get('mimic', {}).get('DBP', {}).get('R2', np.nan),
                'mimic_dbp_rmse': exp['results'].get('mimic', {}).get('DBP', {}).get('RMSE', np.nan),
                'eicu_dbp_r2': exp['results'].get('eicu', {}).get('DBP', {}).get('R2', np.nan),
                'eicu_dbp_rmse': exp['results'].get('eicu', {}).get('DBP', {}).get('RMSE', np.nan),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Calculate degradation
        if 'mimic_sbp_rmse' in df.columns and 'eicu_sbp_rmse' in df.columns:
            df['sbp_degradation_%'] = ((df['eicu_sbp_rmse'] - df['mimic_sbp_rmse']) /
                                       df['mimic_sbp_rmse'] * 100)
            df['dbp_degradation_%'] = ((df['eicu_dbp_rmse'] - df['mimic_dbp_rmse']) /
                                       df['mimic_dbp_rmse'] * 100)

        return df


# ============================================================================
# ENHANCED DATA EXTRACTION WITH BP MODALITY TRACKING
# ============================================================================

class MIMICDataExtractorRevised:
    """Enhanced data extractor with BP measurement modality tracking"""

    def __init__(self, project_id: str = 'your-gcp-project-id'):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.logger = logging.getLogger(__name__)

    def extract_patient_cohort(self, max_patients: int = 10000):
        """Extract patient cohort with inclusion criteria"""
        query = f"""
        WITH patient_cohort AS (
            SELECT DISTINCT
                p.subject_id,
                p.gender,
                DATETIME_DIFF(adm.admittime, p.dob, YEAR) as age,
                adm.hadm_id,
                icu.icustay_id,
                icu.first_careunit,
                icu.los as icu_los,
                adm.insurance,
                adm.language,
                adm.ethnicity
            FROM `physionet-data.mimiciii_clinical.patients` p
            INNER JOIN `physionet-data.mimiciii_clinical.admissions` adm
                ON p.subject_id = adm.subject_id
            INNER JOIN `physionet-data.mimiciii_clinical.icustays` icu
                ON adm.hadm_id = icu.hadm_id
            WHERE 
                DATETIME_DIFF(adm.admittime, p.dob, YEAR) BETWEEN 18 AND 89
                AND icu.los > 1  -- ICU stay > 24 hours
        )
        SELECT * FROM patient_cohort
        LIMIT {max_patients}
        """

        self.logger.info(f"Extracting patient cohort (max {max_patients})...")
        df = self.client.query(query).to_dataframe()
        self.logger.info(f"Extracted {len(df)} patients")
        return df

    def extract_bp_data_with_modality(self, icustay_ids: List[int]):
        """Extract BP data WITH measurement modality (invasive vs non-invasive)"""

        # MIMIC-III item IDs for BP measurements
        # Non-invasive: 442, 443, 220050, 220051, 220179, 220180
        # Invasive: 51, 6701, 220052, 225312 (arterial line)

        query = f"""
        WITH bp_measurements AS (
            SELECT
                ce.icustay_id,
                ce.charttime,
                ce.itemid,
                SAFE_CAST(ce.value AS FLOAT64) as value,
                CASE 
                    -- Invasive (arterial line) measurements
                    WHEN ce.itemid IN (51, 6701, 220052, 225312, 220074, 220075) THEN 'invasive'
                    -- Non-invasive measurements  
                    WHEN ce.itemid IN (442, 443, 220050, 220051, 220179, 220180) THEN 'noninvasive'
                    ELSE 'unknown'
                END as modality,
                CASE
                    WHEN ce.itemid IN (51, 442, 220050, 220179) THEN 'SBP'
                    WHEN ce.itemid IN (8368, 443, 220051, 220180) THEN 'DBP'
                    WHEN ce.itemid IN (6701, 220052, 220074) THEN 'MAP'
                    ELSE NULL
                END as bp_type
            FROM `physionet-data.mimiciii_clinical.chartevents` ce
            WHERE ce.icustay_id IN UNNEST(@icustay_ids)
                AND ce.itemid IN (
                    -- SBP items
                    51, 442, 220050, 220179,
                    -- DBP items
                    8368, 443, 220051, 220180,
                    -- MAP items (for reference)
                    6701, 220052, 220074,
                    -- Additional arterial items
                    220075, 225312
                )
                AND ce.value IS NOT NULL
        )

        SELECT
            icustay_id,
            AVG(CASE WHEN bp_type = 'SBP' AND value > 0 AND value BETWEEN 70 AND 250 THEN value END) as sbp_mean,
            STDDEV(CASE WHEN bp_type = 'SBP' AND value > 0 AND value BETWEEN 70 AND 250 THEN value END) as sbp_std,
            AVG(CASE WHEN bp_type = 'DBP' AND value > 0 AND value BETWEEN 30 AND 150 THEN value END) as dbp_mean,
            STDDEV(CASE WHEN bp_type = 'DBP' AND value > 0 AND value BETWEEN 30 AND 150 THEN value END) as dbp_std,
            -- Modality proportions
            SUM(CASE WHEN modality = 'invasive' AND value > 0 THEN 1 ELSE 0 END) as invasive_count,
            SUM(CASE WHEN modality = 'noninvasive' AND value > 0 THEN 1 ELSE 0 END) as noninvasive_count,
            COUNT(CASE WHEN value > 0 THEN 1 END) as total_measurements
        FROM bp_measurements
        WHERE bp_type IS NOT NULL AND value > 0
        GROUP BY icustay_id
        HAVING COUNT(DISTINCT bp_type) >= 1  -- At least SBP or DBP
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("icustay_ids", "INT64", icustay_ids)
            ]
        )

        self.logger.info("Extracting BP data with modality information...")
        df = self.client.query(query, job_config=job_config).to_dataframe()

        # Calculate modality dominance
        df['invasive_proportion'] = df['invasive_count'] / df['total_measurements']
        df['noninvasive_proportion'] = df['noninvasive_count'] / df['total_measurements']
        df['dominant_modality'] = df.apply(
            lambda x: 'invasive' if x['invasive_proportion'] > 0.5 else
            ('noninvasive' if x['noninvasive_proportion'] > 0.5 else 'mixed'),
            axis=1
        )

        self.logger.info(f"BP modality distribution:")
        self.logger.info(f"  Invasive dominant: {(df['dominant_modality'] == 'invasive').sum()}")
        self.logger.info(f"  Non-invasive dominant: {(df['dominant_modality'] == 'noninvasive').sum()}")
        self.logger.info(f"  Mixed: {(df['dominant_modality'] == 'mixed').sum()}")

        return df

    def extract_vital_signs_features(self, icustay_ids: List[int]):
        """Extract vital signs features"""
        query = f"""
        WITH vitals AS (
            SELECT
                ce.icustay_id,
                ce.itemid,
                SAFE_CAST(ce.value AS FLOAT64) as value
            FROM `physionet-data.mimiciii_clinical.chartevents` ce
            WHERE ce.icustay_id IN UNNEST(@icustay_ids)
                AND ce.itemid IN (
                    211, 220045,  -- Heart rate
                    618, 220210,  -- Respiratory rate
                    646, 220277,  -- SpO2
                    223761, 678  -- Temperature (F)
                )
                AND ce.value IS NOT NULL
        )

        SELECT
            icustay_id,
            -- Heart rate
            AVG(CASE WHEN itemid IN (211, 220045) AND value > 0 THEN value END) as heart_rate_mean,
            STDDEV(CASE WHEN itemid IN (211, 220045) AND value > 0 THEN value END) as heart_rate_std,
            -- Respiratory rate
            AVG(CASE WHEN itemid IN (618, 220210) AND value > 0 THEN value END) as resp_rate_mean,
            STDDEV(CASE WHEN itemid IN (618, 220210) AND value > 0 THEN value END) as resp_rate_std,
            -- SpO2
            AVG(CASE WHEN itemid IN (646, 220277) AND value > 0 THEN value END) as spo2_mean,
            STDDEV(CASE WHEN itemid IN (646, 220277) AND value > 0 THEN value END) as spo2_std,
            -- Temperature (convert to C if needed)
            AVG(CASE WHEN itemid IN (223761, 678) AND value > 0 THEN 
                CASE WHEN value > 50 THEN (value - 32) * 5/9 ELSE value END
            END) as temperature_mean,
            STDDEV(CASE WHEN itemid IN (223761, 678) AND value > 0 THEN 
                CASE WHEN value > 50 THEN (value - 32) * 5/9 ELSE value END
            END) as temperature_std,
            -- Measurement frequency
            COUNT(CASE WHEN value > 0 THEN 1 END) as measurement_frequency
        FROM vitals
        WHERE value > 0
        GROUP BY icustay_id
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("icustay_ids", "INT64", icustay_ids)
            ]
        )

        self.logger.info("Extracting vital signs...")
        df = self.client.query(query, job_config=job_config).to_dataframe()

        # Calculate derived features
        df['heart_rate_cv'] = df['heart_rate_std'] / (df['heart_rate_mean'] + 1e-6)

        return df

    def extract_lab_features(self, icustay_ids: List[int]):
        """Extract laboratory features"""
        query = f"""
        WITH labs AS (
            SELECT
                ie.icustay_id,
                le.itemid,
                SAFE_CAST(le.valuenum AS FLOAT64) as value
            FROM `physionet-data.mimiciii_clinical.labevents` le
            INNER JOIN `physionet-data.mimiciii_clinical.icustays` ie
                ON le.hadm_id = ie.hadm_id
            WHERE ie.icustay_id IN UNNEST(@icustay_ids)
                AND le.valuenum IS NOT NULL
                AND le.itemid IN (
                    50983, -- Sodium
                    50971, -- Potassium
                    50902, -- Chloride
                    50882, -- Bicarbonate
                    50912, -- Creatinine
                    51006, -- BUN
                    51222, -- Hemoglobin
                    51301, -- WBC
                    51265, -- Platelets
                    50931, -- Glucose
                    50813  -- Lactate
                )
        )

        SELECT
            icustay_id,
            AVG(CASE WHEN itemid = 50983 AND value > 0 THEN value END) as sodium,
            AVG(CASE WHEN itemid = 50971 AND value > 0 THEN value END) as potassium,
            AVG(CASE WHEN itemid = 50902 AND value > 0 THEN value END) as chloride,
            AVG(CASE WHEN itemid = 50882 AND value > 0 THEN value END) as bicarbonate,
            AVG(CASE WHEN itemid = 50912 AND value > 0 THEN value END) as creatinine,
            AVG(CASE WHEN itemid = 51006 AND value > 0 THEN value END) as bun,
            AVG(CASE WHEN itemid = 51222 AND value > 0 THEN value END) as hemoglobin,
            AVG(CASE WHEN itemid = 51301 AND value > 0 THEN value END) as wbc,
            AVG(CASE WHEN itemid = 51265 AND value > 0 THEN value END) as platelet,
            AVG(CASE WHEN itemid = 50931 AND value > 0 THEN value END) as glucose,
            AVG(CASE WHEN itemid = 50813 AND value > 0 THEN value END) as lactate
        FROM labs
        GROUP BY icustay_id
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("icustay_ids", "INT64", icustay_ids)
            ]
        )

        self.logger.info("Extracting lab values...")
        df = self.client.query(query, job_config=job_config).to_dataframe()
        return df

    def extract_medication_features(self, icustay_ids: List[int]):
        """Extract medication features"""
        query = f"""
        WITH meds AS (
            SELECT
                icustay_id,
                itemid
            FROM `physionet-data.mimiciii_clinical.inputevents_cv`
            WHERE icustay_id IN UNNEST(@icustay_ids)

            UNION ALL

            SELECT
                icustay_id,
                itemid
            FROM `physionet-data.mimiciii_clinical.inputevents_mv`
            WHERE icustay_id IN UNNEST(@icustay_ids)
        )

        SELECT
            icustay_id,
            -- Vasopressors
            COUNT(DISTINCT CASE WHEN itemid IN (30047, 30120, 221906, 221289) THEN itemid END) as vasopressor_count,
            -- Beta blockers
            MAX(CASE WHEN itemid IN (225151, 225147) THEN 1 ELSE 0 END) as beta_blocker,
            -- Diuretics
            MAX(CASE WHEN itemid IN (225154, 225792) THEN 1 ELSE 0 END) as diuretic,
            -- ACE inhibitors
            MAX(CASE WHEN itemid IN (225149) THEN 1 ELSE 0 END) as ace_inhibitor,
            -- ARBs
            MAX(CASE WHEN itemid IN (225150) THEN 1 ELSE 0 END) as arb,
            -- Calcium channel blockers
            MAX(CASE WHEN itemid IN (225155) THEN 1 ELSE 0 END) as calcium_channel_blocker
        FROM meds
        GROUP BY icustay_id
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("icustay_ids", "INT64", icustay_ids)
            ]
        )

        self.logger.info("Extracting medication data...")
        df = self.client.query(query, job_config=job_config).to_dataframe()
        return df


# ============================================================================
# eICU DATA EXTRACTION
# ============================================================================

class eICUDataExtractor:
    """Extract data from eICU with proper feature alignment"""

    def __init__(self, project_id: str = 'your-gcp-project-id'):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.logger = logging.getLogger(__name__)

    def extract_patient_cohort(self, max_patients: int = 5000):
        """Extract eICU patient cohort"""
        query = f"""
        WITH patient_cohort AS (
            SELECT DISTINCT
                pt.patientunitstayid,
                pt.age,
                pt.gender,
                pt.hospitalid,
                pt.wardid,
                pt.unittype,
                pt.unitstaytype,
                CAST(pt.unitdischargeoffset AS FLOAT64) / (24 * 60) as icu_los_days,
                h.numbedscategory,
                h.teachingstatus,
                h.region
            FROM `physionet-data.eicu_crd.patient` pt
            LEFT JOIN `physionet-data.eicu_crd.hospital` h
                ON pt.hospitalid = h.hospitalid
            WHERE 
                pt.age != ''
                AND SAFE_CAST(pt.age AS INT64) BETWEEN 18 AND 89
                AND CAST(pt.unitdischargeoffset AS FLOAT64) / (24 * 60) > 1  -- > 24 hours
        )
        SELECT * FROM patient_cohort
        LIMIT {max_patients}
        """

        self.logger.info(f"Extracting eICU cohort (max {max_patients})...")
        df = self.client.query(query).to_dataframe()

        # Parse age (eICU stores as string)
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df.dropna(subset=['age'])

        self.logger.info(f"Extracted {len(df)} eICU patients")
        return df

    def extract_bp_data_with_modality(self, patientunitstay_ids: List[int]):
        """Extract BP data from eICU with modality"""
        query = f"""
        WITH bp_measurements AS (
            SELECT
                patientunitstayid,
                observationoffset,
                SAFE_CAST(systemicsystolic AS FLOAT64) as sbp,
                SAFE_CAST(systemicdiastolic AS FLOAT64) as dbp
            FROM `physionet-data.eicu_crd.vitalperiodic`
            WHERE patientunitstayid IN UNNEST(@patientunitstay_ids)
        )

        SELECT
            patientunitstayid,
            AVG(CASE WHEN sbp > 0 AND sbp BETWEEN 70 AND 250 THEN sbp END) as sbp_mean,
            STDDEV(CASE WHEN sbp > 0 AND sbp BETWEEN 70 AND 250 THEN sbp END) as sbp_std,
            AVG(CASE WHEN dbp > 0 AND dbp BETWEEN 30 AND 150 THEN dbp END) as dbp_mean,
            STDDEV(CASE WHEN dbp > 0 AND dbp BETWEEN 30 AND 150 THEN dbp END) as dbp_std,
            0 as invasive_count,
            0 as noninvasive_count,
            COUNT(*) as total_measurements
        FROM bp_measurements
        WHERE (sbp > 0 OR dbp > 0)
        GROUP BY patientunitstayid
        HAVING COUNT(*) >= 2  -- At least 2 measurements
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("patientunitstay_ids", "INT64", patientunitstay_ids)
            ]
        )

        self.logger.info("Extracting eICU BP data...")
        df = self.client.query(query, job_config=job_config).to_dataframe()

        # Calculate modality proportions
        df['invasive_proportion'] = df['invasive_count'] / df['total_measurements']
        df['noninvasive_proportion'] = df['noninvasive_count'] / df['total_measurements']
        df['dominant_modality'] = df.apply(
            lambda x: 'invasive' if x['invasive_proportion'] > 0.5 else
            ('noninvasive' if x['noninvasive_proportion'] > 0.5 else 'mixed'),
            axis=1
        )

        return df

    def extract_vital_signs_features(self, patientunitstay_ids: List[int]):
        """Extract vital signs from eICU"""
        query = f"""
        WITH vitals AS (
            SELECT
                patientunitstayid,
                SAFE_CAST(heartrate AS FLOAT64) as heartrate,
                SAFE_CAST(respiration AS FLOAT64) as respiration,
                SAFE_CAST(sao2 AS FLOAT64) as spo2,
                SAFE_CAST(temperature AS FLOAT64) as temperature
            FROM `physionet-data.eicu_crd.vitalperiodic`
            WHERE patientunitstayid IN UNNEST(@patientunitstay_ids)
        )

        SELECT
            patientunitstayid,
            AVG(CASE WHEN heartrate > 0 THEN heartrate END) as heart_rate_mean,
            STDDEV(CASE WHEN heartrate > 0 THEN heartrate END) as heart_rate_std,
            AVG(CASE WHEN respiration > 0 THEN respiration END) as resp_rate_mean,
            STDDEV(CASE WHEN respiration > 0 THEN respiration END) as resp_rate_std,
            AVG(CASE WHEN spo2 > 0 THEN spo2 END) as spo2_mean,
            STDDEV(CASE WHEN spo2 > 0 THEN spo2 END) as spo2_std,
            AVG(CASE WHEN temperature > 0 THEN temperature END) as temperature_mean,
            STDDEV(CASE WHEN temperature > 0 THEN temperature END) as temperature_std,
            COUNT(CASE WHEN heartrate > 0 OR respiration > 0 OR spo2 > 0 THEN 1 END) as measurement_frequency
        FROM vitals
        GROUP BY patientunitstayid
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("patientunitstay_ids", "INT64", patientunitstay_ids)
            ]
        )

        df = self.client.query(query, job_config=job_config).to_dataframe()
        df['heart_rate_cv'] = df['heart_rate_std'] / (df['heart_rate_mean'] + 1e-6)

        return df

    def extract_lab_features(self, patientunitstay_ids: List[int]):
        """Extract lab features from eICU"""
        query = f"""
        WITH labs AS (
            SELECT
                patientunitstayid,
                labname,
                SAFE_CAST(labresult AS FLOAT64) as labresult
            FROM `physionet-data.eicu_crd.lab`
            WHERE patientunitstayid IN UNNEST(@patientunitstay_ids)
                AND labresult IS NOT NULL
        )

        SELECT
            patientunitstayid,
            AVG(CASE WHEN labname = 'sodium' AND labresult > 0 THEN labresult END) as sodium,
            AVG(CASE WHEN labname = 'potassium' AND labresult > 0 THEN labresult END) as potassium,
            AVG(CASE WHEN labname = 'chloride' AND labresult > 0 THEN labresult END) as chloride,
            AVG(CASE WHEN labname = 'HCO3' AND labresult > 0 THEN labresult END) as bicarbonate,
            AVG(CASE WHEN labname = 'creatinine' AND labresult > 0 THEN labresult END) as creatinine,
            AVG(CASE WHEN labname = 'BUN' AND labresult > 0 THEN labresult END) as bun,
            AVG(CASE WHEN labname = 'Hgb' AND labresult > 0 THEN labresult END) as hemoglobin,
            AVG(CASE WHEN labname = 'WBC x 1000' AND labresult > 0 THEN labresult END) as wbc,
            AVG(CASE WHEN labname = 'platelets x 1000' AND labresult > 0 THEN labresult END) as platelet,
            AVG(CASE WHEN labname = 'glucose' AND labresult > 0 THEN labresult END) as glucose,
            AVG(CASE WHEN labname = 'lactate' AND labresult > 0 THEN labresult END) as lactate
        FROM labs
        GROUP BY patientunitstayid
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("patientunitstay_ids", "INT64", patientunitstay_ids)
            ]
        )

        df = self.client.query(query, job_config=job_config).to_dataframe()
        return df


# ============================================================================
# FEATURE PROCESSOR WITH HARMONIZATION
# ============================================================================

class FeatureProcessor:
    """Process and harmonize features across datasets"""

    def __init__(self, feature_set: str = 'intersection_16'):
        self.feature_set = feature_set
        self.selected_features = FeatureSetConfig.get_feature_set(feature_set)
        self.logger = logging.getLogger(__name__)

    def process_mimic_features(self, cohort, bp, vitals, labs, meds) -> pd.DataFrame:
        """Process MIMIC-III features"""
        # Merge all data
        df = cohort.copy()

        for data in [bp, vitals, labs, meds]:
            df = df.merge(data, on='icustay_id', how='left')

        # Add temporal features
        df['hr_lability_events'] = self._calculate_lability_events(df, 'heart_rate_std', 'heart_rate_mean')

        # Targets
        df['target_sbp'] = df['sbp_mean']
        df['target_dbp'] = df['dbp_mean']

        # Keep only valid patients
        df = df.dropna(subset=['target_sbp', 'target_dbp'])

        # Gender encoding
        df['gender'] = df['gender'].map({'M': 0, 'F': 1})

        self.logger.info(f"Processed MIMIC features: {df.shape}")
        return df

    def process_eicu_features(self, cohort, bp, vitals, labs) -> pd.DataFrame:
        """Process eICU features"""
        df = cohort.copy()

        # Rename ID column
        df = df.rename(columns={'patientunitstayid': 'patient_id'})

        for data in [bp, vitals, labs]:
            data_renamed = data.rename(columns={'patientunitstayid': 'patient_id'})
            df = df.merge(data_renamed, on='patient_id', how='left')

        # Add temporal features
        df['hr_lability_events'] = self._calculate_lability_events(df, 'heart_rate_std', 'heart_rate_mean')

        # Targets
        df['target_sbp'] = df['sbp_mean']
        df['target_dbp'] = df['dbp_mean']

        df = df.dropna(subset=['target_sbp', 'target_dbp'])

        # Gender encoding
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

        self.logger.info(f"Processed eICU features: {df.shape}")
        return df

    def _calculate_lability_events(self, df, std_col, mean_col):
        """Calculate lability events (>20% changes)"""
        if std_col in df.columns and mean_col in df.columns:
            return ((df[std_col] / (df[mean_col] + 1e-6)) > 0.2).astype(int)
        return 0

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Select feature set and separate X, y"""
        # Get intersection of selected features and available features
        available_features = [f for f in self.selected_features if f in df.columns]
        missing_features = [f for f in self.selected_features if f not in df.columns]

        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")

        X = df[available_features].copy()
        y_sbp = df['target_sbp']
        y_dbp = df['target_dbp']

        self.logger.info(f"Selected {len(available_features)} features for {self.feature_set}")

        return X, y_sbp, y_dbp


# ============================================================================
# IMPUTATION STRATEGIES
# ============================================================================

class ImputationStrategy:
    """Multiple imputation strategies with convergence tracking"""

    def __init__(self, method: str = 'median', random_state: int = 42):
        self.method = method
        self.random_state = random_state
        self.imputer = None
        self.convergence_history = []
        self.logger = logging.getLogger(__name__)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training data"""
        if self.method == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif self.method == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        elif self.method == 'mice':
            self.imputer = IterativeImputer(
                random_state=self.random_state,
                max_iter=50,
                tol=1e-4,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

        X_imputed = self.imputer.fit_transform(X_train)
        X_imputed_df = pd.DataFrame(X_imputed, columns=X_train.columns, index=X_train.index)

        self.logger.info(f"Applied {self.method} imputation")

        return X_imputed_df

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Transform test data"""
        if self.imputer is None:
            raise ValueError("Imputer not fitted. Call fit_transform first.")

        X_imputed = self.imputer.transform(X_test)
        X_imputed_df = pd.DataFrame(X_imputed, columns=X_test.columns, index=X_test.index)

        return X_imputed_df

    def validate_imputation_quality(self, X_original: pd.DataFrame, X_imputed: pd.DataFrame):
        """Validate imputation quality with KS tests"""
        results = {}

        for col in X_original.columns:
            observed = X_original[col].dropna()
            imputed_mask = X_original[col].isna()

            if imputed_mask.sum() > 0 and len(observed) > 0:
                imputed_values = X_imputed.loc[imputed_mask, col]
                ks_stat, p_value = ks_2samp(observed, imputed_values)
                results[col] = {'ks_statistic': ks_stat, 'p_value': p_value}

        return results


# ============================================================================
# MODEL ENSEMBLE
# ============================================================================

class BPPredictionModel:
    """Ensemble model for BP prediction"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.meta_model = None
        self.scaler = None
        self.logger = logging.getLogger(__name__)

    def create_base_models(self):
        """Create base models for ensemble"""
        self.models = {
            'gb': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            ),
            'xgb': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
        }

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Fit ensemble model"""
        self.create_base_models()

        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns)

        # Train base models
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            model.fit(X_scaled, y_train)

        # Create stacked ensemble
        estimators = [(name, model) for name, model in self.models.items()]
        self.meta_model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5
        )
        self.meta_model.fit(X_scaled, y_train)

        self.logger.info("Model training complete")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble"""
        X_scaled = self.scaler.transform(X_test)
        X_scaled = pd.DataFrame(X_scaled, columns=X_test.columns)

        # Get predictions from primary GB model
        y_pred_gb = self.models['gb'].predict(X_scaled)

        # Get predictions from stacked ensemble
        y_pred_stacked = self.meta_model.predict(X_scaled)

        # Blend predictions
        y_pred = 0.5 * y_pred_gb + 0.5 * y_pred_stacked

        return y_pred

    def predict_quantiles(self, X_test: pd.DataFrame, quantiles=[0.1, 0.9]):
        """Predict with uncertainty quantification"""
        X_scaled = self.scaler.transform(X_test)
        X_scaled = pd.DataFrame(X_scaled, columns=X_test.columns)

        # Get predictions from all base models
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict(X_scaled))

        predictions = np.array(predictions)

        # Calculate quantiles
        quantile_preds = {}
        for q in quantiles:
            quantile_preds[q] = np.quantile(predictions, q, axis=0)

        return quantile_preds


# ============================================================================
# RECALIBRATION METHODS
# ============================================================================

class RecalibrationModule:
    """Recalibration methods for domain adaptation"""

    def __init__(self):
        self.calibrators = {}
        self.logger = logging.getLogger(__name__)

    def fit_linear_recalibration(self, y_pred_calib, y_true_calib, name='linear'):
        """Fit linear recalibration: y_true = a * y_pred + b"""
        calibrator = Ridge(alpha=1.0)
        calibrator.fit(y_pred_calib.reshape(-1, 1), y_true_calib)
        self.calibrators[name] = calibrator
        self.logger.info(f"Fitted {name} recalibration")

    def fit_isotonic_recalibration(self, y_pred_calib, y_true_calib, name='isotonic'):
        """Fit isotonic (non-parametric) recalibration"""
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_pred_calib, y_true_calib)
        self.calibrators[name] = calibrator
        self.logger.info(f"Fitted {name} recalibration")

    def apply_recalibration(self, y_pred, method='linear'):
        """Apply recalibration"""
        if method not in self.calibrators:
            raise ValueError(f"Calibrator {method} not fitted")

        calibrator = self.calibrators[method]

        if method == 'linear':
            y_recal = calibrator.predict(y_pred.reshape(-1, 1))
        else:  # isotonic
            y_recal = calibrator.predict(y_pred)

        return y_recal


# ============================================================================
# DOMAIN ADAPTATION
# ============================================================================

class DomainAdaptation:
    """Covariate shift reweighting for domain adaptation"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.domain_classifier = None
        self.weights = None
        self.logger = logging.getLogger(__name__)

    def compute_importance_weights(self, X_source: pd.DataFrame, X_target_calib: pd.DataFrame):
        """Compute importance weights: P(target) / P(source)"""
        # Create domain labels
        y_domain = np.concatenate([
            np.zeros(len(X_source)),
            np.ones(len(X_target_calib))
        ])

        X_combined = pd.concat([X_source, X_target_calib], ignore_index=True)

        # Train domain classifier
        self.domain_classifier = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.domain_classifier.fit(X_combined, y_domain)

        # Compute propensity scores
        p_target = self.domain_classifier.predict_proba(X_source)[:, 1]

        # Importance weights
        weights = p_target / (1 - p_target + 1e-6)

        # Clip extreme weights
        weights = np.clip(weights, 0.1, 10.0)

        self.weights = weights
        self.logger.info(f"Computed importance weights. Range: [{weights.min():.2f}, {weights.max():.2f}]")

        return weights

    def fit_reweighted_model(self, model, X_train, y_train, weights):
        """Fit model with importance weights"""
        model.fit(X_train, y_train, sample_weight=weights)
        return model


# ============================================================================
# FEATURE DISTRIBUTION ANALYZER
# ============================================================================

class FeatureDistributionAnalyzer:
    """Analyze feature distribution shifts between datasets"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compute_distribution_shifts(self, X_source: pd.DataFrame, X_target: pd.DataFrame):
        """Compute distribution shift metrics"""
        shifts = []

        for col in X_source.columns:
            if col not in X_target.columns:
                continue

            source_vals = X_source[col].dropna()
            target_vals = X_target[col].dropna()

            if len(source_vals) == 0 or len(target_vals) == 0:
                continue

            # Wasserstein distance
            w_dist = wasserstein_distance(source_vals, target_vals)

            # KS test
            ks_stat, p_value = ks_2samp(source_vals, target_vals)

            # Mean/std comparison
            source_mean = source_vals.mean()
            target_mean = target_vals.mean()
            source_std = source_vals.std()
            target_std = target_vals.std()

            shifts.append({
                'feature': col,
                'wasserstein_distance': w_dist,
                'ks_statistic': ks_stat,
                'ks_pvalue': p_value,
                'source_mean': source_mean,
                'target_mean': target_mean,
                'source_std': source_std,
                'target_std': target_std,
                'mean_shift_%': abs(target_mean - source_mean) / (source_mean + 1e-6) * 100
            })

        df_shifts = pd.DataFrame(shifts).sort_values('wasserstein_distance', ascending=False)

        self.logger.info(f"Analyzed distribution shifts for {len(df_shifts)} features")

        return df_shifts

    def plot_feature_distributions(self, X_source, X_target, top_n=8, output_dir='figures'):
        """Plot distributions of features with largest shifts"""
        os.makedirs(output_dir, exist_ok=True)

        shifts = self.compute_distribution_shifts(X_source, X_target)
        top_features = shifts.head(top_n)['feature'].tolist()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for idx, feature in enumerate(top_features):
            ax = axes[idx]

            source_vals = X_source[feature].dropna()
            target_vals = X_target[feature].dropna()

            ax.hist(source_vals, bins=30, alpha=0.5, label='MIMIC', density=True)
            ax.hist(target_vals, bins=30, alpha=0.5, label='eICU', density=True)
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.set_title(
                f'{feature}\n(W={shifts.loc[shifts["feature"] == feature, "wasserstein_distance"].values[0]:.3f})')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_distribution_shifts.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved feature distribution plot to {output_dir}")


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix='') -> Dict:
    """Compute comprehensive evaluation metrics"""
    metrics = {
        f'{prefix}RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
        f'{prefix}R2': r2_score(y_true, y_pred)
    }

    # Clinical accuracy thresholds
    within_5 = (np.abs(y_true - y_pred) <= 5).mean() * 100
    within_10 = (np.abs(y_true - y_pred) <= 10).mean() * 100
    within_15 = (np.abs(y_true - y_pred) <= 15).mean() * 100

    metrics[f'{prefix}Within_5mmHg_%'] = within_5
    metrics[f'{prefix}Within_10mmHg_%'] = within_10
    metrics[f'{prefix}Within_15mmHg_%'] = within_15

    return metrics


# ============================================================================
# STRATIFIED ANALYSIS
# ============================================================================

class StratifiedAnalyzer:
    """Perform stratified analysis by hospital characteristics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_by_modality(self, df, y_pred_sbp, y_pred_dbp):
        """Analyze performance by BP measurement modality"""
        if 'dominant_modality' not in df.columns:
            self.logger.warning("No modality information available")
            return None

        results = []

        for modality in df['dominant_modality'].unique():
            mask = df['dominant_modality'] == modality

            if mask.sum() < 10:  # Skip if too few samples
                continue

            sbp_metrics = compute_metrics(
                df.loc[mask, 'target_sbp'].values,
                y_pred_sbp[mask],
                prefix=''
            )

            dbp_metrics = compute_metrics(
                df.loc[mask, 'target_dbp'].values,
                y_pred_dbp[mask],
                prefix=''
            )

            results.append({
                'modality': modality,
                'n_samples': mask.sum(),
                'sbp_rmse': sbp_metrics['RMSE'],
                'sbp_r2': sbp_metrics['R2'],
                'dbp_rmse': dbp_metrics['RMSE'],
                'dbp_r2': dbp_metrics['R2']
            })

        return pd.DataFrame(results)

    def analyze_by_hospital_characteristics(self, df, y_pred_sbp, y_pred_dbp):
        """Analyze performance by hospital characteristics (eICU only)"""
        if 'region' not in df.columns and 'teachingstatus' not in df.columns:
            self.logger.warning("No hospital characteristic information available")
            return None

        results = []

        # By region
        if 'region' in df.columns:
            for region in df['region'].unique():
                if pd.isna(region):
                    continue
                mask = df['region'] == region
                if mask.sum() < 20:
                    continue

                sbp_metrics = compute_metrics(df.loc[mask, 'target_sbp'].values, y_pred_sbp[mask], '')
                dbp_metrics = compute_metrics(df.loc[mask, 'target_dbp'].values, y_pred_dbp[mask], '')

                results.append({
                    'stratification': 'region',
                    'category': region,
                    'n_samples': mask.sum(),
                    'sbp_rmse': sbp_metrics['RMSE'],
                    'sbp_r2': sbp_metrics['R2'],
                    'dbp_rmse': dbp_metrics['RMSE'],
                    'dbp_r2': dbp_metrics['R2']
                })

        # By teaching status
        if 'teachingstatus' in df.columns:
            for status in df['teachingstatus'].unique():
                if pd.isna(status):
                    continue
                mask = df['teachingstatus'] == status
                if mask.sum() < 20:
                    continue

                sbp_metrics = compute_metrics(df.loc[mask, 'target_sbp'].values, y_pred_sbp[mask], '')
                dbp_metrics = compute_metrics(df.loc[mask, 'target_dbp'].values, y_pred_dbp[mask], '')

                results.append({
                    'stratification': 'teaching_status',
                    'category': status,
                    'n_samples': mask.sum(),
                    'sbp_rmse': sbp_metrics['RMSE'],
                    'sbp_r2': sbp_metrics['R2'],
                    'dbp_rmse': dbp_metrics['RMSE'],
                    'dbp_r2': dbp_metrics['R2']
                })

        if len(results) == 0:
            return None

        return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================

class VisualizationModule:
    """Generate all figures for the paper"""

    def __init__(self, output_dir='figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def plot_bland_altman(self, y_true, y_pred, title, filename):
        """Create Bland-Altman plot"""
        mean = (y_true + y_pred) / 2
        diff = y_true - y_pred

        mean_diff = diff.mean()
        std_diff = diff.std()

        plt.figure(figsize=(8, 6))
        plt.scatter(mean, diff, alpha=0.5, s=10)
        plt.axhline(mean_diff, color='red', linestyle='--', label=f'Bias: {mean_diff:.2f} mmHg')
        plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle=':',
                    label=f'95% LoA: [{mean_diff - 1.96 * std_diff:.2f}, {mean_diff + 1.96 * std_diff:.2f}]')
        plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle=':')
        plt.xlabel('Mean BP (mmHg)')
        plt.ylabel('Difference (True - Predicted) mmHg')
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_ablation_results(self, tracker: ExperimentTracker):
        """Plot ablation study results"""
        df = tracker.generate_comparison_table()

        # Filter for relevant experiments
        df = df[df['dataset'] == 'eicu']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # SBP RMSE
        ax = axes[0, 0]
        ax.barh(df['feature_set'], df['eicu_sbp_rmse'])
        ax.set_xlabel('SBP RMSE (mmHg)')
        ax.set_title('SBP Prediction Error by Feature Set')
        ax.grid(axis='x', alpha=0.3)

        # SBP R²
        ax = axes[0, 1]
        ax.barh(df['feature_set'], df['eicu_sbp_r2'])
        ax.set_xlabel('SBP R²')
        ax.set_title('SBP R² by Feature Set')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)

        # DBP RMSE
        ax = axes[1, 0]
        ax.barh(df['feature_set'], df['eicu_dbp_rmse'])
        ax.set_xlabel('DBP RMSE (mmHg)')
        ax.set_title('DBP Prediction Error by Feature Set')
        ax.grid(axis='x', alpha=0.3)

        # DBP R²
        ax = axes[1, 1]
        ax.barh(df['feature_set'], df['eicu_dbp_r2'])
        ax.set_xlabel('DBP R²')
        ax.set_title('DBP R² by Feature Set')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_recalibration_impact(self, results_dict: Dict):
        """Plot impact of recalibration methods"""
        methods = list(results_dict.keys())
        sbp_rmse = [results_dict[m]['sbp']['RMSE'] for m in methods]
        sbp_r2 = [results_dict[m]['sbp']['R2'] for m in methods]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(methods, sbp_rmse)
        axes[0].set_ylabel('SBP RMSE (mmHg)')
        axes[0].set_title('Recalibration Impact on RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(methods, sbp_r2)
        axes[1].set_ylabel('SBP R²')
        axes[1].set_title('Recalibration Impact on R²')
        axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'recalibration_impact.png', dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# TABLE GENERATION
# ============================================================================

class TableGenerator:
    """Generate all tables for the paper"""

    def __init__(self, output_dir='tables'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_feature_mapping_table(self):
        """Generate Supplementary Table S1: Feature Mapping"""
        feature_config = FeatureSetConfig()

        all_features = feature_config.get_feature_set('all_28')
        common_features = feature_config.COMMON_FEATURES_16

        rows = []
        for feature in all_features:
            rows.append({
                'Feature': feature,
                'MIMIC_Available': 'Yes',
                'eICU_Available': 'Yes' if feature in common_features else 'No',
                'Feature_Type': self._classify_feature_type(feature),
                'Unit': self._get_feature_unit(feature)
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'table_s1_feature_mapping.csv', index=False)

        self.logger.info("Generated Table S1: Feature Mapping")
        return df

    def _classify_feature_type(self, feature):
        """Classify feature into category"""
        if any(x in feature for x in ['heart_rate', 'resp_rate', 'spo2', 'temperature']):
            return 'Vital Sign'
        elif any(
                x in feature for x in ['sodium', 'potassium', 'creatinine', 'bun', 'glucose', 'lactate', 'hemoglobin']):
            return 'Laboratory'
        elif any(x in feature for x in ['vasopressor', 'beta_blocker', 'diuretic', 'ace', 'arb', 'calcium']):
            return 'Medication'
        elif feature in ['age', 'gender']:
            return 'Demographic'
        else:
            return 'Other'

    def _get_feature_unit(self, feature):
        """Get unit for feature"""
        units = {
            'heart_rate': 'bpm',
            'resp_rate': 'breaths/min',
            'spo2': '%',
            'temperature': '°C',
            'age': 'years',
            'sodium': 'mEq/L',
            'potassium': 'mEq/L',
            'creatinine': 'mg/dL',
            'bun': 'mg/dL',
            'glucose': 'mg/dL',
            'lactate': 'mmol/L',
            'hemoglobin': 'g/dL'
        }

        for key, unit in units.items():
            if key in feature:
                return unit
        return '-'

    def generate_ablation_table(self, tracker: ExperimentTracker):
        """Generate Table III: Ablation Study Results"""
        df = tracker.generate_comparison_table()

        # Format for publication
        df_formatted = df[[
            'exp_id', 'feature_set', 'imputation',
            'mimic_sbp_r2', 'mimic_sbp_rmse',
            'eicu_sbp_r2', 'eicu_sbp_rmse',
            'sbp_degradation_%'
        ]].copy()

        df_formatted = df_formatted.round(3)
        df_formatted.to_csv(self.output_dir / 'table_3_ablation_study.csv', index=False)

        self.logger.info("Generated Table III: Ablation Study")
        return df_formatted

    def generate_stratified_analysis_table(self, stratified_results: pd.DataFrame):
        """Generate Table V: Stratified Analysis"""
        if stratified_results is None:
            self.logger.warning("No stratified results available")
            return None

        stratified_results.to_csv(self.output_dir / 'table_5_stratified_analysis.csv', index=False)
        self.logger.info("Generated Table V: Stratified Analysis")
        return stratified_results

    def generate_distribution_shift_table(self, shift_df: pd.DataFrame):
        """Generate supplementary table on distribution shifts"""
        shift_df_formatted = shift_df.head(20).round(3)  # Top 20 features with largest shifts
        shift_df_formatted.to_csv(self.output_dir / 'table_s3_distribution_shifts.csv', index=False)
        self.logger.info("Generated Table S3: Distribution Shifts")
        return shift_df_formatted


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

class ComprehensiveExperimentRunner:
    """Run all experiments systematically"""

    def __init__(self, config: Dict):
        self.config = config
        self.tracker = ExperimentTracker(config.get('output_dir', 'results'))
        self.logger = logging.getLogger(__name__)

        # Storage for intermediate results
        self.mimic_data = {}
        self.eicu_data = {}
        self.models = {}

    def run_all_experiments(self):
        """Run complete experimental pipeline"""

        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING COMPREHENSIVE EXPERIMENTAL PIPELINE")
        self.logger.info("=" * 80)

        # Phase 1: Data Extraction
        self.logger.info("\nPhase 1: Data Extraction")
        self._extract_all_data()

        # Phase 2: Baseline Experiment (Current Approach - 22 features)
        self.logger.info("\nPhase 2: Baseline Experiment (Current 22 Features)")
        self._run_baseline_experiment()

        # Phase 3: Critical Experiment - Intersection Only (16 features)
        self.logger.info("\nPhase 3: CRITICAL EXPERIMENT - Intersection Only (16 Features)")
        self._run_intersection_only_experiment()

        # Phase 4: Ablation Studies
        self.logger.info("\nPhase 4: Ablation Studies")
        self._run_ablation_studies()

        # Phase 5: Imputation Sensitivity
        self.logger.info("\nPhase 5: Imputation Sensitivity Analysis")
        self._run_imputation_sensitivity()

        # Phase 6: Recalibration Experiments
        self.logger.info("\nPhase 6: Recalibration Experiments")
        self._run_recalibration_experiments()

        # Phase 7: Domain Adaptation
        self.logger.info("\nPhase 7: Domain Adaptation (Covariate Shift Reweighting)")
        self._run_domain_adaptation()

        # Phase 8: Stratified Analysis
        self.logger.info("\nPhase 8: Stratified Analysis")
        self._run_stratified_analysis()

        # Phase 9: Feature Distribution Analysis
        self.logger.info("\nPhase 9: Feature Distribution Analysis")
        self._analyze_feature_distributions()

        # Phase 10: Generate Tables and Figures
        self.logger.info("\nPhase 10: Generating Tables and Figures")
        self._generate_all_outputs()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 80)

    def _extract_all_data(self):
        """Extract data from both databases"""
        # MIMIC-III
        mimic_extractor = MIMICDataExtractorRevised(
            project_id=self.config['data_extraction']['project_id']
        )

        cohort = mimic_extractor.extract_patient_cohort(
            max_patients=self.config['data_extraction']['max_patients']
        )
        ids = cohort['icustay_id'].tolist()

        bp = mimic_extractor.extract_bp_data_with_modality(ids)
        vitals = mimic_extractor.extract_vital_signs_features(ids)
        labs = mimic_extractor.extract_lab_features(ids)
        meds = mimic_extractor.extract_medication_features(ids)

        processor = FeatureProcessor(feature_set='all_28')
        mimic_full = processor.process_mimic_features(cohort, bp, vitals, labs, meds)

        # Store
        self.mimic_data['full'] = mimic_full
        self.mimic_data['bp_modality'] = bp[['icustay_id', 'dominant_modality',
                                             'invasive_proportion', 'noninvasive_proportion']]

        self.logger.info(f"✓ MIMIC-III data extracted: {mimic_full.shape}")

        # eICU
        if not self.config.get('skip_eicu', False):
            eicu_extractor = eICUDataExtractor(
                project_id=self.config['data_extraction']['project_id']
            )

            cohort_eicu = eicu_extractor.extract_patient_cohort(
                max_patients=self.config['data_extraction']['max_eicu_patients']
            )
            ids_eicu = cohort_eicu['patientunitstayid'].tolist()

            bp_eicu = eicu_extractor.extract_bp_data_with_modality(ids_eicu)
            vitals_eicu = eicu_extractor.extract_vital_signs_features(ids_eicu)
            labs_eicu = eicu_extractor.extract_lab_features(ids_eicu)

            processor_eicu = FeatureProcessor(feature_set='all_28')
            eicu_full = processor_eicu.process_eicu_features(cohort_eicu, bp_eicu, vitals_eicu, labs_eicu)

            # Store
            self.eicu_data['full'] = eicu_full
            bp_modality_eicu = bp_eicu[['patientunitstayid', 'dominant_modality',
                                                     'invasive_proportion', 'noninvasive_proportion']].copy()
            bp_modality_eicu = bp_modality_eicu.rename(columns={'patientunitstayid': 'patient_id'})
            self.eicu_data['bp_modality'] = bp_modality_eicu
            self.eicu_data['cohort'] = cohort_eicu

            self.logger.info(f"✓ eICU data extracted: {eicu_full.shape}")

    def _run_baseline_experiment(self):
        """Run baseline experiment with current 22-feature approach"""
        exp_id = 'EXP_BASELINE_22FEAT'

        processor = FeatureProcessor(feature_set='current_22')
        X, y_sbp, y_dbp = processor.select_features(self.mimic_data['full'])

        # Train-test split
        X_train, X_test, y_sbp_train, y_sbp_test, y_dbp_train, y_dbp_test = train_test_split(
            X, y_sbp, y_dbp, test_size=0.2, random_state=42
        )

        # Imputation
        imputer = ImputationStrategy(method='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Train models
        model_sbp = BPPredictionModel(random_state=42)
        model_dbp = BPPredictionModel(random_state=42)

        model_sbp.fit(X_train_imp, y_sbp_train)
        model_dbp.fit(X_train_imp, y_dbp_train)

        # Evaluate on MIMIC
        y_pred_sbp_mimic = model_sbp.predict(X_test_imp)
        y_pred_dbp_mimic = model_dbp.predict(X_test_imp)

        mimic_results = {
            'SBP': compute_metrics(y_sbp_test.values, y_pred_sbp_mimic, ''),
            'DBP': compute_metrics(y_dbp_test.values, y_pred_dbp_mimic, '')
        }

        # Evaluate on eICU
        eicu_results = None
        if 'full' in self.eicu_data:
            X_eicu, y_sbp_eicu, y_dbp_eicu = processor.select_features(self.eicu_data['full'])

            # Handle missing features with median imputation from MIMIC
            X_eicu_aligned = self._align_features(X_eicu, X_train_imp)
            X_eicu_imp = imputer.transform(X_eicu_aligned)

            y_pred_sbp_eicu = model_sbp.predict(X_eicu_imp)
            y_pred_dbp_eicu = model_dbp.predict(X_eicu_imp)

            eicu_results = {
                'SBP': compute_metrics(y_sbp_eicu.values, y_pred_sbp_eicu, ''),
                'DBP': compute_metrics(y_dbp_eicu.values, y_pred_dbp_eicu, '')
            }

        # Log experiment
        self.tracker.log_experiment(
            exp_id=exp_id,
            config={'feature_set': 'current_22', 'imputation_method': 'median', 'dataset': 'both'},
            results={'mimic': mimic_results, 'eicu': eicu_results}
        )

        # Store models
        self.models['baseline'] = {'sbp': model_sbp, 'dbp': model_dbp, 'imputer': imputer}

        self.logger.info(f"✓ Baseline experiment complete: {exp_id}")
        self.logger.info(f"  MIMIC SBP R²={mimic_results['SBP']['R2']:.3f}, RMSE={mimic_results['SBP']['RMSE']:.2f}")
        if eicu_results:
            self.logger.info(f"  eICU SBP R²={eicu_results['SBP']['R2']:.3f}, RMSE={eicu_results['SBP']['RMSE']:.2f}")

    def _run_intersection_only_experiment(self):
        """CRITICAL: Run intersection-only experiment (16 features, no imputation)"""
        exp_id = 'EXP_INTERSECTION_16FEAT'

        processor = FeatureProcessor(feature_set='intersection_16')
        X, y_sbp, y_dbp = processor.select_features(self.mimic_data['full'])

        # Remove any rows with missing values in intersection features
        valid_mask = X.notna().all(axis=1)
        X = X[valid_mask]
        y_sbp = y_sbp[valid_mask]
        y_dbp = y_dbp[valid_mask]

        self.logger.info(f"After removing missing values: {len(X)} samples")

        # Train-test split
        X_train, X_test, y_sbp_train, y_sbp_test, y_dbp_train, y_dbp_test = train_test_split(
            X, y_sbp, y_dbp, test_size=0.2, random_state=42
        )

        # NO IMPUTATION - these are the 16 common features
        # But apply minimal imputation if any NaN remain
        imputer = ImputationStrategy(method='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Train models
        model_sbp = BPPredictionModel(random_state=42)
        model_dbp = BPPredictionModel(random_state=42)

        model_sbp.fit(X_train_imp, y_sbp_train)
        model_dbp.fit(X_train_imp, y_dbp_train)

        # Evaluate on MIMIC
        y_pred_sbp_mimic = model_sbp.predict(X_test_imp)
        y_pred_dbp_mimic = model_dbp.predict(X_test_imp)

        mimic_results = {
            'SBP': compute_metrics(y_sbp_test.values, y_pred_sbp_mimic, ''),
            'DBP': compute_metrics(y_dbp_test.values, y_pred_dbp_mimic, '')
        }

        # Evaluate on eICU - CRITICAL: Use only intersection features
        eicu_results = None
        if 'full' in self.eicu_data:
            X_eicu, y_sbp_eicu, y_dbp_eicu = processor.select_features(self.eicu_data['full'])

            # Remove missing values
            valid_mask_eicu = X_eicu.notna().all(axis=1)
            X_eicu = X_eicu[valid_mask_eicu]
            y_sbp_eicu = y_sbp_eicu[valid_mask_eicu]
            y_dbp_eicu = y_dbp_eicu[valid_mask_eicu]

            X_eicu_imp = imputer.transform(X_eicu)

            y_pred_sbp_eicu = model_sbp.predict(X_eicu_imp)
            y_pred_dbp_eicu = model_dbp.predict(X_eicu_imp)

            eicu_results = {
                'SBP': compute_metrics(y_sbp_eicu.values, y_pred_sbp_eicu, ''),
                'DBP': compute_metrics(y_dbp_eicu.values, y_pred_dbp_eicu, '')
            }

        # Log experiment
        self.tracker.log_experiment(
            exp_id=exp_id,
            config={'feature_set': 'intersection_16', 'imputation_method': 'none', 'dataset': 'both'},
            results={'mimic': mimic_results, 'eicu': eicu_results}
        )

        # Store models
        self.models['intersection'] = {'sbp': model_sbp, 'dbp': model_dbp, 'imputer': imputer}

        self.logger.info(f"✓ CRITICAL EXPERIMENT complete: {exp_id}")
        self.logger.info(f"  MIMIC SBP R²={mimic_results['SBP']['R2']:.3f}, RMSE={mimic_results['SBP']['RMSE']:.2f}")
        if eicu_results:
            self.logger.info(f"  eICU SBP R²={eicu_results['SBP']['R2']:.3f}, RMSE={eicu_results['SBP']['RMSE']:.2f}")
            self.logger.info(f"  *** This is the KEY result that determines if fundamental barriers exist ***")

    def _run_ablation_studies(self):
        """Run ablation studies with different feature subsets"""
        feature_sets = ['no_meds_17', 'no_labs_16']

        for feature_set in feature_sets:
            exp_id = f'EXP_ABLATION_{feature_set.upper()}'

            processor = FeatureProcessor(feature_set=feature_set)
            X, y_sbp, y_dbp = processor.select_features(self.mimic_data['full'])

            X_train, X_test, y_sbp_train, y_sbp_test, y_dbp_train, y_dbp_test = train_test_split(
                X, y_sbp, y_dbp, test_size=0.2, random_state=42
            )

            imputer = ImputationStrategy(method='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            model_sbp = BPPredictionModel(random_state=42)
            model_sbp.fit(X_train_imp, y_sbp_train)

            y_pred_sbp_mimic = model_sbp.predict(X_test_imp)
            mimic_results = {'SBP': compute_metrics(y_sbp_test.values, y_pred_sbp_mimic, '')}

            # eICU
            eicu_results = None
            if 'full' in self.eicu_data:
                X_eicu, y_sbp_eicu, _ = processor.select_features(self.eicu_data['full'])
                X_eicu_aligned = self._align_features(X_eicu, X_train_imp)
                X_eicu_imp = imputer.transform(X_eicu_aligned)

                y_pred_sbp_eicu = model_sbp.predict(X_eicu_imp)
                eicu_results = {'SBP': compute_metrics(y_sbp_eicu.values, y_pred_sbp_eicu, '')}

            self.tracker.log_experiment(
                exp_id=exp_id,
                config={'feature_set': feature_set, 'imputation_method': 'median', 'dataset': 'both'},
                results={'mimic': mimic_results, 'eicu': eicu_results}
            )

            self.logger.info(f"✓ Ablation experiment: {exp_id}")

    def _run_imputation_sensitivity(self):
        """Test different imputation methods"""
        imputation_methods = ['median', 'knn', 'mice']

        for method in imputation_methods:
            exp_id = f'EXP_IMPUTATION_{method.upper()}_22FEAT'

            processor = FeatureProcessor(feature_set='current_22')
            X, y_sbp, y_dbp = processor.select_features(self.mimic_data['full'])

            X_train, X_test, y_sbp_train, y_sbp_test, y_dbp_train, y_dbp_test = train_test_split(
                X, y_sbp, y_dbp, test_size=0.2, random_state=42
            )

            imputer = ImputationStrategy(method=method)
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            model_sbp = BPPredictionModel(random_state=42)
            model_sbp.fit(X_train_imp, y_sbp_train)

            y_pred_sbp_mimic = model_sbp.predict(X_test_imp)
            mimic_results = {'SBP': compute_metrics(y_sbp_test.values, y_pred_sbp_mimic, '')}

            # eICU
            eicu_results = None
            if 'full' in self.eicu_data:
                X_eicu, y_sbp_eicu, _ = processor.select_features(self.eicu_data['full'])
                X_eicu_aligned = self._align_features(X_eicu, X_train_imp)
                X_eicu_imp = imputer.transform(X_eicu_aligned)

                y_pred_sbp_eicu = model_sbp.predict(X_eicu_imp)
                eicu_results = {'SBP': compute_metrics(y_sbp_eicu.values, y_pred_sbp_eicu, '')}

            self.tracker.log_experiment(
                exp_id=exp_id,
                config={'feature_set': 'current_22', 'imputation_method': method, 'dataset': 'both'},
                results={'mimic': mimic_results, 'eicu': eicu_results}
            )

            self.logger.info(f"✓ Imputation sensitivity: {method}")

    def _run_recalibration_experiments(self):
        """Run recalibration experiments"""
        if 'full' not in self.eicu_data:
            self.logger.warning("Skipping recalibration - no eICU data")
            return

        # Use intersection model
        if 'intersection' not in self.models:
            self.logger.warning("Intersection model not available, using baseline")
            model_sbp = self.models['baseline']['sbp']
            imputer = self.models['baseline']['imputer']
            feature_set = 'current_22'
        else:
            model_sbp = self.models['intersection']['sbp']
            imputer = self.models['intersection']['imputer']
            feature_set = 'intersection_16'

        processor = FeatureProcessor(feature_set=feature_set)
        X_eicu, y_sbp_eicu, _ = processor.select_features(self.eicu_data['full'])

        # Split eICU into calibration and test
        X_eicu_calib, X_eicu_test, y_eicu_calib, y_eicu_test = train_test_split(
            X_eicu, y_sbp_eicu, test_size=0.8, random_state=42
        )

        # Impute
        X_eicu_calib_imp = imputer.transform(X_eicu_calib)
        X_eicu_test_imp = imputer.transform(X_eicu_test)

        # Get base predictions
        y_pred_calib = model_sbp.predict(X_eicu_calib_imp)
        y_pred_test = model_sbp.predict(X_eicu_test_imp)

        # No recalibration baseline
        results_no_recal = compute_metrics(y_eicu_test.values, y_pred_test, '')

        # Linear recalibration
        recal = RecalibrationModule()
        recal.fit_linear_recalibration(y_pred_calib, y_eicu_calib.values)
        y_pred_linear = recal.apply_recalibration(y_pred_test, method='linear')
        results_linear = compute_metrics(y_eicu_test.values, y_pred_linear, '')

        # Isotonic recalibration
        recal.fit_isotonic_recalibration(y_pred_calib, y_eicu_calib.values)
        y_pred_isotonic = recal.apply_recalibration(y_pred_test, method='isotonic')
        results_isotonic = compute_metrics(y_eicu_test.values, y_pred_isotonic, '')

        # Store results
        self.recalibration_results = {
            'no_recalibration': {'sbp': results_no_recal},
            'linear': {'sbp': results_linear},
            'isotonic': {'sbp': results_isotonic}
        }

        self.logger.info("✓ Recalibration experiments complete")
        self.logger.info(f"  No recal: R²={results_no_recal['R2']:.3f}")
        self.logger.info(f"  Linear:   R²={results_linear['R2']:.3f}")
        self.logger.info(f"  Isotonic: R²={results_isotonic['R2']:.3f}")

    def _run_domain_adaptation(self):
        """Run domain adaptation with covariate shift reweighting"""
        if 'full' not in self.eicu_data:
            self.logger.warning("Skipping domain adaptation - no eICU data")
            return

        processor = FeatureProcessor(feature_set='intersection_16')
        X_mimic, y_sbp_mimic, _ = processor.select_features(self.mimic_data['full'])
        X_eicu, y_sbp_eicu, _ = processor.select_features(self.eicu_data['full'])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_mimic, y_sbp_mimic, test_size=0.2, random_state=42
        )

        # Small eICU calibration set
        X_eicu_calib, X_eicu_test, y_eicu_calib, y_eicu_test = train_test_split(
            X_eicu, y_sbp_eicu, test_size=0.8, random_state=42
        )

        # Impute
        imputer = ImputationStrategy(method='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_eicu_test_imp = imputer.transform(X_eicu_test)
        X_eicu_calib_imp = imputer.transform(X_eicu_calib)

        # Domain adaptation
        domain_adapter = DomainAdaptation()
        weights = domain_adapter.compute_importance_weights(X_train_imp, X_eicu_calib_imp)

        # Train reweighted model
        model_reweighted = BPPredictionModel(random_state=42)
        model_reweighted.create_base_models()

        # Manually fit with weights (using GB model as primary)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train_imp)
        model_reweighted.models['gb'].fit(X_scaled, y_train.values, sample_weight=weights)
        model_reweighted.scaler = scaler

        # Evaluate
        y_pred_reweighted = model_reweighted.models['gb'].predict(scaler.transform(X_eicu_test_imp))
        results_reweighted = compute_metrics(y_eicu_test.values, y_pred_reweighted, '')

        self.domain_adaptation_results = {'sbp': results_reweighted}

        self.logger.info("✓ Domain adaptation complete")
        self.logger.info(f"  Reweighted model R²={results_reweighted['R2']:.3f}")

    def _run_stratified_analysis(self):
        """Run stratified analysis by hospital characteristics"""
        if 'full' not in self.eicu_data:
            self.logger.warning("Skipping stratified analysis - no eICU data")
            return

        # Use intersection model predictions
        model_sbp = self.models['intersection']['sbp']
        model_dbp = self.models['intersection']['dbp']
        imputer = self.models['intersection']['imputer']

        processor = FeatureProcessor(feature_set='intersection_16')
        X_eicu, y_sbp_eicu, y_dbp_eicu = processor.select_features(self.eicu_data['full'])

        X_eicu_imp = imputer.transform(X_eicu)

        y_pred_sbp = model_sbp.predict(X_eicu_imp)
        y_pred_dbp = model_dbp.predict(X_eicu_imp)

        # Merge with cohort data for stratification
        df_eicu = self.eicu_data['full'].copy()
        df_eicu = df_eicu.loc[X_eicu.index]  # Align indices

        analyzer = StratifiedAnalyzer()

        # By modality
        df_eicu_with_modality = df_eicu.merge(
            self.eicu_data['bp_modality'],
            left_on='patient_id',
            right_on='patient_id',
            how='left'
        )

        results_modality = analyzer.analyze_by_modality(df_eicu_with_modality, y_pred_sbp, y_pred_dbp)

        # By hospital characteristics
        results_hospital = analyzer.analyze_by_hospital_characteristics(df_eicu, y_pred_sbp, y_pred_dbp)

        self.stratified_results = {
            'modality': results_modality,
            'hospital': results_hospital
        }

        self.logger.info("✓ Stratified analysis complete")

    def _analyze_feature_distributions(self):
        """Analyze feature distribution shifts"""
        if 'full' not in self.eicu_data:
            self.logger.warning("Skipping feature distribution analysis - no eICU data")
            return

        processor = FeatureProcessor(feature_set='intersection_16')
        X_mimic, _, _ = processor.select_features(self.mimic_data['full'])
        X_eicu, _, _ = processor.select_features(self.eicu_data['full'])

        analyzer = FeatureDistributionAnalyzer()
        shift_df = analyzer.compute_distribution_shifts(X_mimic, X_eicu)

        # Save
        shift_df.to_csv(f"{self.config.get('output_dir', 'results')}/feature_distribution_shifts.csv", index=False)

        # Plot
        analyzer.plot_feature_distributions(
            X_mimic, X_eicu,
            output_dir=self.config.get('figure_dir', 'figures')
        )

        self.distribution_shifts = shift_df

        self.logger.info("✓ Feature distribution analysis complete")

    def _generate_all_outputs(self):
        """Generate all tables and figures"""
        # Tables
        table_gen = TableGenerator(output_dir=self.config.get('table_dir', 'tables'))
        table_gen.generate_feature_mapping_table()
        table_gen.generate_ablation_table(self.tracker)

        if hasattr(self, 'stratified_results'):
            if self.stratified_results['modality'] is not None:
                table_gen.generate_stratified_analysis_table(self.stratified_results['modality'])

        if hasattr(self, 'distribution_shifts'):
            table_gen.generate_distribution_shift_table(self.distribution_shifts)

        # Figures
        viz = VisualizationModule(output_dir=self.config.get('figure_dir', 'figures'))
        viz.plot_ablation_results(self.tracker)

        if hasattr(self, 'recalibration_results'):
            viz.plot_recalibration_impact(self.recalibration_results)

        self.logger.info("✓ Generated all tables and figures")

    def _align_features(self, X_target: pd.DataFrame, X_source: pd.DataFrame) -> pd.DataFrame:
        """Align target features with source features (fill missing with median)"""
        X_aligned = X_target.copy()

        for col in X_source.columns:
            if col not in X_aligned.columns:
                # Fill with median from source
                X_aligned[col] = X_source[col].median()

        # Ensure same column order
        X_aligned = X_aligned[X_source.columns]

        return X_aligned


# ============================================================================
# MAIN
# ============================================================================

def setup_logging_revised(log_dir='results'):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"bp_revision_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive BP Prediction Pipeline - Revision')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    parser.add_argument('--max-patients', type=int, default=10000, help='Max MIMIC patients')
    parser.add_argument('--max-eicu', type=int, default=5000, help='Max eICU patients')
    parser.add_argument('--skip-eicu', action='store_true', help='Skip eICU validation')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging_revised(args.output_dir)

    # Load config
    config = {
        'data_extraction': {
            'project_id': 'your-gcp-project-id',
            'max_patients': args.max_patients,
            'max_eicu_patients': args.max_eicu
        },
        'skip_eicu': args.skip_eicu,
        'output_dir': args.output_dir,
        'figure_dir': 'figures',
        'table_dir': 'tables'
    }

    if args.config:
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)

    # Run comprehensive experiments
    runner = ComprehensiveExperimentRunner(config)

    try:
        runner.run_all_experiments()
        logger.info("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
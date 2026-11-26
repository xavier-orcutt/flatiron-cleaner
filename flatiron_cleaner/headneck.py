import pandas as pd
import numpy as np
import logging
import re 
from typing import Optional
from .general import DataProcessorGeneral

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorHeadNeck(DataProcessorGeneral):
    GROUP_STAGE_MAPPING = {
        'IV': 'IV_NOS',  
        'IVA': 'IV_locoregional',
        'IVB': 'IV_locoregional',
        'IVC': 'IV_metastatic',
        'III': 'III',
        'IIIA': 'III',
        'IIIB': 'III',
        'IIIC': 'III',
        'II': 'II',
        'IIA': 'II',
        'IIB': 'II',
        'IIC': 'II',
        'I': 'I',
        'IA': 'I',
        'IB': 'I',
        '0': '0',  
        'Unknown/not documented': 'unknown'
    }

    HPV_STATUS_MAPPING = {
        'HPV positive': 'positive',
        'HPV negative': 'negative',
        'NaN': 'unknown',
        'Unknown': 'unknown',
        'Results pending': 'unknown',
        'Unsuccessful/indeterminate test': 'unknown',
        'HPV equivocal': 'unknown'
    }

    PDL1_CPS_MAPPING = {
        '0': 1, 
        '<1': 2,
        '1': 3, 
        '2-4': 4,
        '5-9': 5,
        '10-19': 6,  
        '20-29': 7, 
        '30-39': 8, 
        '40-49': 9, 
        '50-59': 10, 
        '60-69': 11, 
        '70-79': 12, 
        '80-89': 13, 
        '90-99': 14,
        '100': 15
    }

    PDL1_PERCENT_STAINING_MAPPING = {
        '0%': 1, 
        '< 1%': 2,
        '1%': 3, 
        '2% - 4%': 4,
        '5% - 9%': 5,
        '10% - 19%': 6,  
        '20% - 29%': 7, 
        '30% - 39%': 8, 
        '40% - 49%': 9, 
        '50% - 59%': 10, 
        '60% - 69%': 11, 
        '70% - 79%': 12, 
        '80% - 89%': 13, 
        '90% - 99%': 14,
        '100%': 15
    }
    
    def __init__(self):
        super().__init__() 

        # head neck-specific attributes
        self.enhanced_df = None
        self.biomarkers_df = None
        self.enahnced_df = None

    def process_enhanced(self,
                         file_path: str,
                         index_date_df: pd.DataFrame,
                         index_date_column: str,
                         drop_stage: bool = True,
                         drop_hpv: bool = True, 
                         drop_treatment: bool = True,
                         drop_dates: bool = True) -> Optional[pd.DataFrame]: 
        """
        Processes Enhanced_AdvHeadNeck.csv to standardize categories, consolidate 
        staging information, and calculate time-based metrics between key clinical events.

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvHeadNeck.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only mortality data for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        drop_stage : bool, default=True
            If True, drops original staging columns (GroupStage) after creating modified versions
        drop_hpv : bool, default=True
            If True, drops original HPV status columns (HPVStatus) after creating modified versions
        drop_treatment : bool, default=True
            If True, drops original surgery (IsPrimarySurgery) and radiation (PrimaryRadiationTherapy) columns after creating modified versions
        drop_dates : bool, default=True
            If True, drops date columns (DiagnosisDate, AdvancedDiagnosisDate, and SurgeryDate) after calculating durations

        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - AdvancedDiagnosticCriteria : category 
                advanced diagnostic criteria; unchanged from source dataframe 
            - GroupStage_mod : category
                consolidated overall staging (0-IV and unknown) at time of first diagnosis
            - PrimarySite : category
                anatomical site of cancer; unchanged from source dataframe
            - SmokingStatus : category
                smoking history; unchanged from source dataframe
            - HPVTested : category 
                testing status for HPV; unchanged from source dataframe
            - HPVStatus_mod : category
                consolidated HPV status
            - received_surgery: Int64
                binary indicator (0/1) for whether patient had surgery prior to index date 
            - received_radiation : Int64
                binary indicator (0/1) for whether patient had radiation prior to index date 
            - had_local_recurrence : Int64
                binary indicator (0/1) for whether patient had local recurrence prior to index date
            - had_distant_recurrence : Int64
                binary indicator (0/1) for whether patient had distant recurrence prior to index date
            - days_diagnosis_to_adv : float
                days from first diagnosis to advanced disease 
            - adv_diagnosis_year : category
                year of advanced diagnosis 
            
            Original staging, HPV, treatment, and date columns retained if respective drop_* parameters = False

        Notes
        -----
        Output handling:
        - Duplicate PatientIDs are logged as warnings if found but reatained in output
        - Processed DataFrame is stored in self.enhanced_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'        

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_AdvHeadNeck.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Select PatientIDs that are included in the index_date_df the merge on 'left'
            df = df[df.PatientID.isin(index_date_df.PatientID)]
            df = pd.merge(
                 df,
                 index_date_df[['PatientID', index_date_column]],
                 on = 'PatientID',
                 how = 'left'
            )
            logging.info(f"Successfully filtered Enhanced_AdvHeadNeck.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Convert date columns
            date_cols = ['DiagnosisDate', 
                         'AdvancedDiagnosisDate',
                         index_date_column,
                         'FirstLocalRecurDate',
                         'FirstDistantRecurDate',
                         'PrimarySurgeryDate',
                         'PrimaryRadiationDate']
            
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])

            # Convert categorical columns
            categorical_cols = ['AdvancedDiagnosisCriteria', 
                                'GroupStage', 
                                'PrimarySite',
                                'SmokingStatus',
                                'HPVTested',
                                'HPVStatus']
        
            df[categorical_cols] = df[categorical_cols].astype('category')

            # Recode stage and HPV status variables using class-level mapping and create new column
            df['GroupStage_mod'] = df['GroupStage'].map(self.GROUP_STAGE_MAPPING).astype('category')
            df['HPVStatus_mod'] = df['HPVStatus'].map(self.HPV_STATUS_MAPPING).astype('category')

            # Drop original stage and HPV variables if specified
            if drop_stage:
                df = df.drop(columns=['GroupStage'])

            if drop_hpv:
                df = df.drop(columns=['HPVStatus'])
            
            # Generate treatment related variables 
            df['received_surgery'] = np.where(df['PrimarySurgeryDate'] <= df[index_date_column], 1, 0)
            df['received_surgery'] = df['received_surgery'].astype('Int64')

            df['received_radiation'] = np.where(df['PrimaryRadiationDate'] <= df[index_date_column], 1, 0)
            df['received_radiation'] = df['received_radiation'].astype('Int64')

            # Drop original treatment variables if specified
            if drop_treatment:
                df = df.drop(columns=['IsPrimarySurgery', 'PrimaryRadiationTherapy'])

            # Generate recurrence variables 
            df['had_local_recurrence'] = np.where(df['FirstLocalRecurDate'] <= df[index_date_column], 1, 0)
            df['had_local_recurrence'] = df['had_local_recurrence'].astype('Int64')

            df['had_distant_recurrence'] = np.where(df['FirstDistantRecurDate'] <= df[index_date_column], 1, 0)
            df['had_distant_recurrence'] = df['had_distant_recurrence'].astype('Int64')

            # Generate time-based variables 
            df['days_diagnosis_to_adv'] = (df['AdvancedDiagnosisDate'] - df['DiagnosisDate']).dt.days
            df['adv_diagnosis_year'] = pd.Categorical(df['AdvancedDiagnosisDate'].dt.year) 
        
            if drop_dates:
                df = df.drop(columns = ['DiagnosisDate', 
                                        'AdvancedDiagnosisDate', 
                                        index_date_column,
                                        'FirstLocalRecurDate',
                                        'FirstDistantRecurDate',
                                        'PrimarySurgeryDate',
                                        'PrimaryRadiationDate'])

            # Check for duplicate PatientIDs
            if len(df) > df['PatientID'].nunique():
                duplicate_ids = df[df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Enhanced_AdvHeadNeck.csv file with final shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            self.enhanced_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvHeadNeck.csv file: {e}")
            return None
        
    def process_biomarkers(self,
                               file_path: str,
                               index_date_df: pd.DataFrame,
                               index_date_column: str, 
                               days_before: Optional[int] = None,
                               days_after: int = 0,
                               pdl1_result_type: str = 'cps',
                               cps_positive_cutoff: str = '1') -> Optional[pd.DataFrame]:
            """
            Processes Enhanced_AdvHeadNeckBiomarkers.csv by determining FGFR and PDL1 status for each patient within a specified time window relative to an index date
    
            Parameters
            ----------
            file_path : str
                Path to Enhanced_AdvHeadNeckBiomarkers.csv file
            index_date_df : pd.DataFrame
                DataFrame containing unique PatientIDs and their corresponding index dates. Only biomarker data for PatientIDs present in this DataFrame will be processed
            index_date_column : str
                Column name in index_date_df containing the index date
            days_before : int | None, optional
                Number of days before the index date to include. Must be >= 0 or None. If None, includes all prior results. Default: None
            days_after : int, optional
                Number of days after the index date to include. Must be >= 0. Default: 0
            pdl1_result_type: str, default = 'cps'
                Type of PD-L1 quantification to return: 
                    - 'cps': Combined Positive Score
                    - 'percent_staining': Tumor Proportion Score (TPS). Less commonly used in head and neck cancers
            cps_positive_cutoff : str, default = '1'
                CPS category at or above which a result counts as PD-L1 positive for PDL1_status, regardless
                of BiomarkerStatus. Must be a key of PDL1_CPS_MAPPING (e.g., '1' for CPS >= 1, '20-29' for CPS >= 20).
                Set to None to classify status from BiomarkerStatus alone
            
            Returns
            -------
            pd.DataFrame or None
                - PatientID : object
                    unique patient identifier
                - PDL1_status : category
                    positive if any test has BiomarkerStatus 'PD-L1 positive' or CPS >= cps_positive_cutoff;
                    negative if no positive tests and any test is 'PD-L1 negative/not detected' or has CPS below the cutoff;
                    otherwise unknown
                - PDL1_cps : category, ordered 
                    if pdl1_result_type = 'cps', returns a patient's maximum CPS across all PD-L1 tests in the window, including negative results (NaN if no recorded CPS or CPS is 'Unknown/not documented')
                - PDL1_percent_staining : category, ordered 
                    if pdl1_result_type = 'percent_staining', returns a patient's maximum percent staining across all PD-L1 tests in the window, including negative results (NaN if no recorded percent staining)
    
            Notes
            ------
            Biomarker cleaning and processing: 
            - PDL1 status is classified per test, then per patient:
                - a test is positive if BiomarkerStatus is 'PD-L1 positive' OR CPS >= cps_positive_cutoff
                - a test is negative if BiomarkerStatus is 'PD-L1 negative/not detected' OR CPS < cps_positive_cutoff
                - patient is 'positive' if any test is positive (ever-positive), 'negative' if any test is negative
                  without positives (only-negative), otherwise 'unknown'
                - CPS takes precedence over a negative BiomarkerStatus (e.g., status negative with CPS 5-9 is positive)
    
            - Missing biomarker data handling:
                - All PatientIDs from index_date_df are included in the output
                - Patients without any biomarker tests will have NaN values for all biomarker columns
                - Missing ResultDate is imputed with SpecimenReceivedDate
    
            Output handling: 
            - Duplicate PatientIDs are logged as warnings if found but retained in output
            - Processed DataFrame is stored in self.biomarkers_df
            """
            # Input validation
            if not isinstance(index_date_df, pd.DataFrame):
                raise ValueError("index_date_df must be a pandas DataFrame")
            if 'PatientID' not in index_date_df.columns:
                raise ValueError("index_date_df must contain a 'PatientID' column")
            if not index_date_column or index_date_column not in index_date_df.columns:
                raise ValueError('index_date_column not found in index_date_df')
            if index_date_df['PatientID'].duplicated().any():
                raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
            
            if days_before is not None:
                if not isinstance(days_before, int) or days_before < 0:
                    raise ValueError("days_before must be a non-negative integer or None")
            if not isinstance(days_after, int) or days_after < 0:
                raise ValueError("days_after must be a non-negative integer")
            if cps_positive_cutoff is not None and cps_positive_cutoff not in self.PDL1_CPS_MAPPING:
                raise ValueError(f"cps_positive_cutoff must be None or one of {list(self.PDL1_CPS_MAPPING)}")
            
            index_date_df = index_date_df.copy()
            # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
            for col in index_date_df.columns:
                if col != 'PatientID':  # Keep PatientID unchanged for merging
                    index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)
    
            # Update index_date_column name
            index_date_column = f'imported_{index_date_column}'
    
            try:
                df = pd.read_csv(file_path)
                logging.info(f"Successfully read Enhanced_AdvHeadNeckBiomarkers.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
    
                df['ResultDate'] = pd.to_datetime(df['ResultDate'])
                df['SpecimenReceivedDate'] = pd.to_datetime(df['SpecimenReceivedDate'])
    
                # Impute missing ResultDate with SpecimenReceivedDate
                df['ResultDate'] = np.where(df['ResultDate'].isna(), df['SpecimenReceivedDate'], df['ResultDate'])
    
                index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])
    
                # Select PatientIDs that are included in the index_date_df the merge on 'left'
                df = df[df.PatientID.isin(index_date_df.PatientID)]
                df = pd.merge(
                     df,
                     index_date_df[['PatientID', index_date_column]],
                     on = 'PatientID',
                     how = 'left'
                )
                logging.info(f"Successfully merged Enhanced_AdvHeadNeckBiomarkers.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
                
                # Create new variable 'index_to_result' that notes difference in days between resulted specimen and index date
                df['index_to_result'] = (df['ResultDate'] - df[index_date_column]).dt.days
                
                # Select biomarkers that fall within desired before and after index date
                if days_before is None:
                    # Only filter for days after
                    df_filtered = df[df['index_to_result'] <= days_after].copy()
                else:
                    # Filter for both before and after
                    df_filtered = df[
                        (df['index_to_result'] <= days_after) & 
                        (df['index_to_result'] >= -days_before)
                    ].copy()
    
                inconsistent = df_filtered[
                    (df_filtered['BiomarkerStatus'] == 'PD-L1 positive') & 
                    (df_filtered['CombinedPositiveScore'].isin(['0', '<1']))
                ]
    
                if len(inconsistent) > 0:
                    inconsistent_ids = inconsistent.PatientID.unique().tolist()
                    logging.warning(f"Found {len(inconsistent)} records (PatientIDs: {inconsistent_ids}) with PD-L1 positive but CPS 0 or <1 - possible data quality issue")
                
                # Replace unknown CPS with empty (used for both status and CPS output)
                df_filtered.loc[df_filtered['CombinedPositiveScore'] == "Unknown/not documented", 'CombinedPositiveScore'] = np.nan
    
                # Process PDL1 status: classify each test, then each patient
                pdl1_tests = df_filtered.query('BiomarkerName == "PDL1"').copy()
                cps_ordinal = pdl1_tests['CombinedPositiveScore'].map(self.PDL1_CPS_MAPPING)
    
                status_positive = pdl1_tests['BiomarkerStatus'].eq('PD-L1 positive')
                status_negative = pdl1_tests['BiomarkerStatus'].eq('PD-L1 negative/not detected')
    
                if cps_positive_cutoff is not None:
                    cutoff = self.PDL1_CPS_MAPPING[cps_positive_cutoff]
                    cps_positive = cps_ordinal.ge(cutoff)   # NaN CPS -> False
                    cps_negative = cps_ordinal.lt(cutoff)   # NaN CPS -> False
                else:
                    cps_positive = pd.Series(False, index = pdl1_tests.index)
                    cps_negative = pd.Series(False, index = pdl1_tests.index)
    
                pdl1_tests['test_positive'] = status_positive | cps_positive
                pdl1_tests['test_negative'] = (status_negative | cps_negative) & ~pdl1_tests['test_positive']
    
                PDL1_df = (
                    pdl1_tests
                    .groupby('PatientID')[['test_positive', 'test_negative']]
                    .any()
                    .assign(PDL1_status = lambda x: np.select(
                        [x['test_positive'], x['test_negative']],
                        ['positive', 'negative'],
                        default = 'unknown'))
                    .reset_index()
                    [['PatientID', 'PDL1_status']]
                )
    
                n_cps_override = int((status_negative & cps_positive).sum())
                if n_cps_override > 0:
                    logging.info(f"{n_cps_override} PD-L1 records with BiomarkerStatus 'PD-L1 negative/not detected' were classified positive based on CPS >= {cps_positive_cutoff}")
    
                # Process PDL1 staining 
                if pdl1_result_type == 'cps': 
                    # Use every PD-L1 row with a recorded CPS, regardless of BiomarkerStatus,
                    # so negative results ('0', '<1') are retained
                    pdl1_rows = df_filtered.query('BiomarkerName == "PDL1"').copy()
                    pdl1_rows['PDL1_cps_ordinal_value'] = pdl1_rows['CombinedPositiveScore'].map(self.PDL1_CPS_MAPPING)
    
                    # Maximum CPS across all qualifying tests in the window
                    PDL1_cps_df = (
                        pdl1_rows
                        .dropna(subset = ['PDL1_cps_ordinal_value'])
                        .groupby('PatientID')['PDL1_cps_ordinal_value']
                        .max()
                        .reset_index()
                    )
                    reverse_pdl1_cps_dict = {v: k for k, v in self.PDL1_CPS_MAPPING.items()}
                    PDL1_cps_df['PDL1_cps'] = PDL1_cps_df['PDL1_cps_ordinal_value'].map(reverse_pdl1_cps_dict)
                    PDL1_cps_df = PDL1_cps_df.drop(columns = ['PDL1_cps_ordinal_value'])
    
                    # Merge dataframes -- start with index_date_df to ensure all PatientIDs are included
                    final_df = index_date_df[['PatientID']].copy()
                    final_df = pd.merge(final_df, PDL1_df, on = 'PatientID', how = 'left')
                    final_df = pd.merge(final_df, PDL1_cps_df, on = 'PatientID', how = 'left')
    
                    final_df['PDL1_status'] = final_df['PDL1_status'].astype('category')
    
                    cps_dtype = pd.CategoricalDtype(
                        categories = ['0', '<1', '1', '2-4', '5-9', '10-19',
                                    '20-29', '30-39', '40-49', '50-59',
                                    '60-69', '70-79', '80-89', '90-99', '100'],
                                    ordered = True
                    )
                    
                    final_df['PDL1_cps'] = final_df['PDL1_cps'].astype(cps_dtype)
                    
                elif pdl1_result_type == 'percent_staining': 
                    # Use every PD-L1 row with a recorded percent staining, regardless of
                    # BiomarkerStatus, so negative results ('0%', '< 1%') are retained
                    pdl1_rows = df_filtered.query('BiomarkerName == "PDL1"').copy()
                    pdl1_rows['PDL1_percent_staining_ordinal_value'] = pdl1_rows['PercentStaining'].map(self.PDL1_PERCENT_STAINING_MAPPING)
    
                    # Maximum percent staining across all qualifying tests in the window
                    PDL1_percent_staining_df = (
                        pdl1_rows
                        .dropna(subset = ['PDL1_percent_staining_ordinal_value'])
                        .groupby('PatientID')['PDL1_percent_staining_ordinal_value']
                        .max()
                        .reset_index()
                    )
                    reverse_pdl1_percent_staining_dict = {v: k for k, v in self.PDL1_PERCENT_STAINING_MAPPING.items()}
                    PDL1_percent_staining_df['PDL1_percent_staining'] = PDL1_percent_staining_df['PDL1_percent_staining_ordinal_value'].map(reverse_pdl1_percent_staining_dict)
                    PDL1_percent_staining_df = PDL1_percent_staining_df.drop(columns = ['PDL1_percent_staining_ordinal_value'])
    
                    # Merge dataframes -- start with index_date_df to ensure all PatientIDs are included
                    final_df = index_date_df[['PatientID']].copy()
                    final_df = pd.merge(final_df, PDL1_df, on = 'PatientID', how = 'left')
                    final_df = pd.merge(final_df, PDL1_percent_staining_df, on = 'PatientID', how = 'left')
    
                    final_df['PDL1_status'] = final_df['PDL1_status'].astype('category')
    
                    percent_staining_dtype = pd.CategoricalDtype(
                        categories = ['0%', '< 1%', '1%', '2% - 4%', '5% - 9%', '10% - 19%',
                                    '20% - 29%', '30% - 39%', '40% - 49%', '50% - 59%',
                                    '60% - 69%', '70% - 79%', '80% - 89%', '90% - 99%', '100%'],
                                    ordered = True
                    )
                    
                    final_df['PDL1_percent_staining'] = final_df['PDL1_percent_staining'].astype(percent_staining_dtype)
    
                else: 
                    raise ValueError("pdl1_result_type must be 'cps' or 'percent_staining'")
    
                # Check for duplicate PatientIDs
                if len(final_df) > final_df['PatientID'].nunique():
                    duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                    logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")
    
                logging.info(f"Successfully processed Enhanced_AdvHeadNeckBiomarkers.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
                self.biomarkers_df = final_df
                return final_df
    
            except Exception as e:
                logging.error(f"Error processing Enhanced_AdvHeadNeckBiomarkers.csv file: {e}")
                return None
        
    def process_mortality(self,
                          file_path: str,
                          index_date_df: pd.DataFrame,
                          index_date_column: str,
                          visit_path: str = None, 
                          telemedicine_path: str = None, 
                          biomarkers_path: str = None,
                          drop_dates: bool = True) -> Optional[pd.DataFrame]:
        """
        Processes Enhanced_Mortality_V2.csv by cleaning data types, calculating time from index date to death/censor, and determining mortality events. 

        Parameters
        ----------
        file_path : str
            Path to Enhanced_Mortality_V2.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only mortality data for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        visit_path : str
            Path to Visit.csv file, using VisitDate to determine last EHR activity date for censored patients
        telemedicine_path : str
            Path to Telemedicine.csv file, using VisitDate to determine last EHR activity date for censored patients
        biomarkers_path : str
            Path to Enhanced_AdvUrothelialBiomarkers.csv file, using SpecimenCollectedDate to determine last EHR activity date for censored patients
        drop_dates : bool, default = True
            If True, drops date columns (index_date_column, DateOfDeath, last_ehr_date)   
        
        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - duration : float
                days from index date to death or censor 
            - event : Int64
                mortality status (1 = death, 0 = censored)

        If drop_dates=False, the DataFrame will also include:
            - {index_date_column} : datetime64
                The index date for each patient
            - DateOfDeath : datetime64
                Date of death (if available)
            - last_ehr_activity : datetime64
                Most recent EHR activity date (if available from supplementary files)
                
        Notes
        ------
        Death date handling:
        - Known death date: 'event' = 1, 'duration' = days from index to death
        - No death date: 'event' = 0, 'duration' = days from index to last EHR activity
        
        Death date imputation for incomplete dates:
        - Missing day: Imputed to 15th of the month
        - Missing month and day: Imputed to July 1st of the year
    
        Censoring logic:
        - Patients without death dates are censored at their last EHR activity
        - Last EHR activity is determined as the maximum date across all provided supplementary files (visit, telemedicine, biomarkers, oral)
        - If no supplementary files are provided or a patient has no activity in supplementary files, duration may be null for censored patients
        
        Output handling: 
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.mortality_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_Mortality_V2.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # When only year is available: Impute to July 1st (mid-year)
            df['DateOfDeath'] = np.where(df['DateOfDeath'].str.len() == 4, df['DateOfDeath'] + '-07-01', df['DateOfDeath'])

            # When only month and year are available: Impute to the 15th day of the month
            df['DateOfDeath'] = np.where(df['DateOfDeath'].str.len() == 7, df['DateOfDeath'] + '-15', df['DateOfDeath'])

            df['DateOfDeath'] = pd.to_datetime(df['DateOfDeath'])

            # Process index dates and merge
            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])
            df = pd.merge(
                index_date_df[['PatientID', index_date_column]],
                df,
                on = 'PatientID',
                how = 'left'
            )
            logging.info(f"Successfully merged Enhanced_Mortality_V2.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
                
            # Create event column
            df['event'] = df['DateOfDeath'].notna().astype('Int64')

            # Initialize final dataframe
            final_df = df.copy()

            # Create a list to store all last activity date dataframes
            patient_last_dates = []

            # Determine last EHR data
            if all(path is None for path in [visit_path, telemedicine_path, biomarkers_path]):
                logging.info("WARNING: At least one of visit_path, telemedicine_path, or biomarkers_path be provided to calculate duration for those with a missing death date")
            else: 
                # Process visit and telemedicine data
                if visit_path is not None or telemedicine_path is not None:
                    visit_dates = []
                    try:
                        if visit_path is not None:
                            df_visit = pd.read_csv(visit_path)
                            df_visit['VisitDate'] = pd.to_datetime(df_visit['VisitDate'])
                            visit_dates.append(df_visit[['PatientID', 'VisitDate']])
                            
                        if telemedicine_path is not None:
                            df_tele = pd.read_csv(telemedicine_path)
                            df_tele['VisitDate'] = pd.to_datetime(df_tele['VisitDate'])
                            visit_dates.append(df_tele[['PatientID', 'VisitDate']])
                        
                        if visit_dates:
                            df_visit_combined = pd.concat(visit_dates)
                            df_visit_max = (
                                df_visit_combined
                                .query("PatientID in @index_date_df.PatientID")
                                .groupby('PatientID')['VisitDate']
                                .max()
                                .to_frame(name='last_visit_date')
                                .reset_index()
                            )
                            patient_last_dates.append(df_visit_max)
                    except Exception as e:
                        logging.error(f"Error processing Visit.csv or Telemedicine.csv: {e}")
                                            
                # Process biomarkers data
                if biomarkers_path is not None:
                    try: 
                        df_biomarkers = pd.read_csv(biomarkers_path)
                        df_biomarkers['SpecimenCollectedDate'] = pd.to_datetime(df_biomarkers['SpecimenCollectedDate'])

                        df_biomarkers_max = (
                            df_biomarkers
                            .query("PatientID in @index_date_df.PatientID")
                            .groupby('PatientID')['SpecimenCollectedDate']
                            .max()
                            .to_frame(name='last_biomarker_date')
                            .reset_index()
                        )
                        patient_last_dates.append(df_biomarkers_max)
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvHeadNeckBiomarkers.csv file: {e}")

                # Combine all last activity dates
                if patient_last_dates:
                    # Start with the first dataframe
                    combined_dates = patient_last_dates[0]
                    
                    # Merge with any additional dataframes
                    for date_df in patient_last_dates[1:]:
                        combined_dates = pd.merge(combined_dates, date_df, on = 'PatientID', how = 'outer')
                    
                    # Calculate the last activity date across all columns
                    date_columns = [col for col in combined_dates.columns if col != 'PatientID']
                    if date_columns:
                        logging.info(f"The following columns {date_columns} are used to calculate the last EHR date")
                        combined_dates['last_ehr_activity'] = combined_dates[date_columns].max(axis=1)
                        single_date = combined_dates[['PatientID', 'last_ehr_activity']]
                        
                        # Merge with the main dataframe
                        final_df = pd.merge(final_df, single_date, on='PatientID', how='left')
     
            # Calculate duration
            if 'last_ehr_activity' in final_df.columns:
                final_df['duration'] = np.where(
                    final_df['event'] == 0, 
                    (final_df['last_ehr_activity'] - final_df[index_date_column]).dt.days, 
                    (final_df['DateOfDeath'] - final_df[index_date_column]).dt.days
                )
                
                # Drop date variables if specified
                if drop_dates:               
                    final_df = final_df.drop(columns=[index_date_column, 'DateOfDeath', 'last_ehr_activity'])
                       
            else: 
                final_df['duration'] = (final_df['DateOfDeath'] - final_df[index_date_column]).dt.days
            
                # Drop date variables if specified
                if drop_dates:               
                    final_df = final_df.drop(columns=[index_date_column, 'DateOfDeath'])
                
            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset=['PatientID'], keep=False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Enhanced_Mortality_V2.csv file with final shape: {final_df.shape} and unique PatientIDs: {final_df['PatientID'].nunique()}. There are {final_df['duration'].isna().sum()} out of {final_df['PatientID'].nunique()} patients with missing duration values")
            self.mortality_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_Mortality_V2.csv file: {e}")
            return None
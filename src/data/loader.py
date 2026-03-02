"""ECG Data Loading Module"""

import os
import ast
import wfdb
import numpy as np
import pandas as pd


class ECGDataLoader:
    """Load and manage ECG dataset from PTB-XL"""
    
    def __init__(self, path_to_data: str):
        """
        Initialize the data loader.
        
        Args:
            path_to_data: Path to the PTB-XL dataset directory
        """
        self.path_to_data = path_to_data
        self.ecg_df = None
        self.scp_df = None
        self.ecg_data = None
    
    def load_metadata(self) -> tuple:
        """
        Load metadata from ECG and SCP files.
        
        Returns:
            tuple: (ECG_df, SCP_df) dataframes
        """
        # Load ECG metadata
        self.ecg_df = pd.read_csv(
            os.path.join(self.path_to_data, 'ptbxl_database.csv'), 
            index_col='ecg_id'
        )
        self.ecg_df.scp_codes = self.ecg_df.scp_codes.apply(lambda x: ast.literal_eval(x))
        self.ecg_df.patient_id = self.ecg_df.patient_id.astype(int)
        self.ecg_df.nurse = self.ecg_df.nurse.astype('Int64')
        self.ecg_df.site = self.ecg_df.site.astype('Int64')
        self.ecg_df.validated_by = self.ecg_df.validated_by.astype('Int64')
        
        # Load SCP statements
        self.scp_df = pd.read_csv(
            os.path.join(self.path_to_data, 'scp_statements.csv'), 
            index_col=0
        )
        self.scp_df = self.scp_df[self.scp_df.diagnostic == 1]
        
        return self.ecg_df, self.scp_df
    
    def add_diagnostic_classes(self):
        """Add diagnostic superclasses to ECG dataframe"""
        def diagnostic_class(scp):
            res = set()
            for k in scp.keys():
                if k in self.scp_df.index:
                    res.add(self.scp_df.loc[k].diagnostic_class)
            return list(res)
        
        self.ecg_df['scp_classes'] = self.ecg_df.scp_codes.apply(diagnostic_class)
    
    def load_raw_data(self, sampling_rate: int = 100) -> np.ndarray:
        """
        Load raw ECG signal data.
        
        Args:
            sampling_rate: Sampling rate (100 or 500 Hz)
        
        Returns:
            np.ndarray: ECG signal data
        """
        if sampling_rate == 100:
            data = [wfdb.rdsamp(os.path.join(self.path_to_data, f)) 
                   for f in self.ecg_df.filename_lr]
        else:
            data = [wfdb.rdsamp(os.path.join(self.path_to_data, f)) 
                   for f in self.ecg_df.filename_hr]
        
        self.ecg_data = np.array([signal for signal, meta in data])
        return self.ecg_data
    
    @property
    def superclasses(self) -> list:
        """Get list of diagnostic superclasses"""
        return ['NORM', 'MI', 'STTC', 'CD', 'HYP']

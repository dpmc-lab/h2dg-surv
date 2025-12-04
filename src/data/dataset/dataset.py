import os
import re
import glob
import h5py
import torch
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from functools import reduce
from typing import Tuple, Any, Dict, Optional, List
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def collate_fn(data):
    """Collate function to batch samples together."""
    # First, unpack the main structure: tensor_tuples (input) and tensor_tuples (target)
    patient_id, tensor_tuples, target = zip(*data)
    # Then, transpose the tensor tuples to get individual tensor lists
    clinical, blood, patho, cdm, h_ids, h_mask, s_ids, s_mask, r_ids, r_mask, lymph, tumor = zip(*tensor_tuples)
    time_to_event, survival_status = zip(*target)

    clinical = torch.stack(clinical, 0)
    blood    = torch.stack(blood,    0)
    patho    = torch.stack(patho,    0)
    cdm      = torch.stack(cdm,      0)
    lymph    = torch.stack(lymph, 0)
    tumor    = torch.stack(tumor, 0)

    h_ids  = torch.vstack(h_ids)
    h_mask = torch.vstack(h_mask)
    s_ids  = torch.vstack(s_ids)
    s_mask = torch.vstack(s_mask)
    r_ids  = torch.vstack(r_ids)
    r_mask = torch.vstack(r_mask)
    
    time_to_event    = torch.stack(time_to_event, 0)
    survival_status  = torch.stack(survival_status, 0)

    input = (clinical, blood, patho, cdm, h_ids, h_mask, s_ids, s_mask, r_ids, r_mask, lymph, tumor)
    target = (time_to_event, survival_status)
    patient_id = torch.stack(patient_id, 0)

    return patient_id, input, target


class Dataset(Dataset):
    """
    HANCOCK Dataset for multimodal survival prediction.
    
    Loads and preprocesses 9 modalities:
    - Clinical data (demographics, smoking, metastasis)
    - Blood data (lab tests with reference ranges)
    - Pathological data (tumor staging, grading, invasion markers)
    - TMA cell density measurements
    - Text data: histories, surgery descriptions, reports (tokenized)
    - WSI features: lymph node, primary tumor (H5 files)
    """
    
    def __init__(
        self,
        max_tokens_history: int,
        max_tokens_surgery: int,
        max_tokens_report: int,
        path_lm: str,
        split: str,
        data_clinical: pd.DataFrame,
        data_blood: pd.DataFrame,
        data_pathological: pd.DataFrame,
        tma_cdm: pd.DataFrame,
        data_root: str = "./data/HANCOCK",
        list_patient_id_sample: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize HANCOCK Dataset.
        
        Args:
            max_tokens_history: Maximum tokens for history text
            max_tokens_surgery: Maximum tokens for surgery description text
            max_tokens_report: Maximum tokens for report text
            path_lm: Path to language model (Bio_ClinicalBERT)
            split: Dataset split name (train/val/test)
            data_root: Root directory of HANCOCK dataset
            list_patient_id_sample: Optional list of patient IDs to include
            data_clinical: pre-processed clinical data
            data_blood: pre-processed blood data
            data_pathological: pre-processed pathological data
            tma_cdm: pre-processed TMA data
        """
        # Device
        self.device = torch.device('cpu')
        
        # Important features
        self.split         = split
        self.path_data     = data_root
        self.list_patient_id_sample = list_patient_id_sample

        self.max_tokens_history = max_tokens_history
        self.max_tokens_surgery = max_tokens_surgery
        self.max_tokens_report  = max_tokens_report
        self.path_lm            = path_lm

        # Prepare samples
        self.data_clinical = data_clinical
        self.data_blood = data_blood
        self.data_pathological = data_pathological
        self.tma_cdm = tma_cdm
        self.prepare_samples()
        
        # Build tokenizer
        self.build_tokenizer()

    def log_dataset(self) -> None:
        """Log dataset information."""
        print(f'\n>>> Details about the {self.split} set:')
        print(f'    > Number of samples: {len(self.samples)}\n')

    def build_tokenizer(self) -> None:
        """Build tokenizer from pretrained language model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_lm, local_files_only=True, do_lower_case=True)

    def prepare_samples(self) -> None:
        # Text data
        self.prepare_accession_text()
        # Image paths
        self.prepare_lymph_node()
        self.prepare_primary_tumor()

        # Filter samples based on patient ID list
        self.samples = list(self.data_clinical.patient_id)
        # Use provided patient list if available, otherwise use all patients
        if self.list_patient_id_sample is not None:
            # Filter to only include patients in the provided list
            valid_patient_ids = [pid for pid in self.list_patient_id_sample if pid in self.samples]
            self.samples = valid_patient_ids

    def read_h5(self, path):
        if path is None:
            data = np.zeros((1, 1024))
        else:
            with h5py.File(path, 'r') as f:
                data = f['features'][:]
        return data

    def get_covariates(self) -> pd.DataFrame:
        """
        Merge all tabular covariates, excluding images and texts.
        
        Returns:
            Merged dataframe with all structured data
        """
        dfs = [
            self.data_clinical,
            self.data_blood,
            self.data_pathological,
            self.tma_cdm,
        ]
        df = reduce(lambda left, 
                             right: pd.merge(left, right, on='patient_id', how='outer'), 
                             dfs)
        return df

    def prepare_primary_tumor(self) -> None:
        SUBFOLDER = os.path.join(self.path_data, f"WSI_PrimaryTumor")
        FILES     = [str(x) for x in list(Path(SUBFOLDER).rglob("*.h5"))]

        patient_dict = {f"{i:03d}": None for i in range(1, 764)}
        pattern      = re.compile(r'(\d{3})')

        for path in FILES:
            filename = os.path.basename(path)
            match = pattern.search(filename)
            if match:
                pid = match.group(1)  # Extracted patient ID
                if patient_dict[pid] is None:
                    patient_dict[pid] = []
                patient_dict[pid].append(path)

        self.primary_tumors_paths = patient_dict

    def prepare_lymph_node(self) -> None:
        SUBFOLDER = os.path.join(self.path_data, f"WSI_LymphNode/h5_files")

        patient_dict = {}
        for i in range(1, 763 + 1):
            patient_id = f"{i:03d}"
            PATH_H5 = Path(os.path.join(SUBFOLDER, f"LymphNode_HE_{patient_id}.h5"))
            if PATH_H5.is_file():
                patient_dict[patient_id] = str(PATH_H5)
            else:
                patient_dict[patient_id] = None
            
        self.lymph_nodes_paths = patient_dict

    def read_txt(self, path):
        try:
            with open(path, 'r') as file:
                txt = file.read()
        except:
            txt = "No report available."

        return txt

    def prepare_accession_text(self) -> None:
        SUBFOLDER = os.path.join(self.path_data, f"TextData")

        path_histories = os.path.join(SUBFOLDER, f"histories_english")
        path_surgeries = os.path.join(SUBFOLDER, f"surgery_descriptions_english")
        path_reports   = os.path.join(SUBFOLDER, f"reports_english")

        self.histories, self.surgeries, self.reports = {}, {}, {}

        for idx in range(1, 764):
            idx = str(idx).zfill(3)
            
            path_history      = os.path.join(path_histories, f"SurgeryReport_History_{idx}.txt")
            path_report       = os.path.join(path_reports, f"SurgeryReport_{idx}.txt")
            path_description  = os.path.join(path_surgeries, f"SurgeryDescriptionEnglish_{idx}.txt")

            self.histories[idx] = self.read_txt(path_history)
            self.surgeries[idx] = self.read_txt(path_report)
            self.reports[idx]   = self.read_txt(path_description)

    def __len__(self) -> int:
        return len(self.samples)

    def png_img_to_tensor(self, path_image: str) -> torch.Tensor:
        """Convert PNG image to tensor and apply transformations"""
        # load apacc images
        img = Image.open(path_image)
        # Convert to float32 for better precision and conserve 3 first channels
        img = np.array(img).astype(np.float32)[:, :, :3] / 255.0
        # apply transforms
        tensor = self.transform(img)
        return tensor

    def get_labels(self, path_img_file: str) -> Tuple[str, torch.Tensor]:
        # extract id of patient
        image_name = os.path.basename(path_img_file)
        # extract label as string
        label_str = self.labels[self.labels.image_name == image_name].label.values[0]
        # create a one-hot tensor
        index         = self.label_to_index[label_str]
        labels        = torch.zeros(self.num_classes)
        labels[index] = 1
        return image_name, labels
        
    def __get_clinical__(self, idx):
        df_patient       = self.data_clinical.loc[self.data_clinical.patient_id == idx]
        df_patient_no_id = df_patient.drop(columns=['patient_id', 'survival_status', 'survival_status_with_cause', 'days_to_last_information'])
        patient_array    = torch.tensor(df_patient_no_id.values.astype(float), dtype=torch.float32).flatten()
        return patient_array
    
    def __get_blood__(self, idx):
        df_patient       = self.data_blood.loc[self.data_blood.patient_id == idx]
        df_patient_no_id = df_patient.drop(columns=['patient_id'])
        patient_array    = torch.tensor(df_patient_no_id.values.astype(float), dtype=torch.float32).flatten()
        return patient_array

    def __get_pathological__(self, idx):
        df_patient       = self.data_pathological.loc[self.data_pathological.patient_id == idx]
        df_patient_no_id = df_patient.drop(columns=['patient_id'])
        patient_array    = torch.tensor(df_patient_no_id.values.astype(float), dtype=torch.float32).flatten()
        return patient_array  

    def __get_tma_cdm__(self, idx):
        df_patient       = self.tma_cdm.loc[self.tma_cdm.patient_id == idx]
        df_patient_no_id = df_patient.drop(columns=['patient_id'])
        patient_array    = torch.tensor(df_patient_no_id.values.astype(float), dtype=torch.float32).flatten()
        return patient_array 

    def __trunc__(self, text, max_tokens):
        encodings = self.tokenizer(text, return_tensors='pt', max_length=max_tokens, padding='max_length', truncation=True)
        ids       = encodings['input_ids']
        mask      = encodings['attention_mask']
        return ids, mask
    
    def __middle_trunc__(self, text):
        encodings = self.tokenizer(text, return_tensors='pt')
        ids       = encodings['input_ids']
        mask      = encodings['attention_mask']
        n_tokens  = ids.shape[-1]

        if n_tokens > 512:
            half       = n_tokens // 2
            ids_trunc  = torch.hstack((ids[:, half-256:half], ids[:, half:half+256]))
            mask_trunc = torch.ones_like(ids_trunc)
            return ids_trunc, mask_trunc
        
        else:
            encodings = self.tokenizer(text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
            ids       = encodings['input_ids']
            mask      = encodings['attention_mask']    
            return ids, mask
        
    def __tokenize__(self, text, max_tokens):
        if max_tokens == 512:
            ids, mask = self.__middle_trunc__(text)
        else:
            ids, mask = self.__trunc__(text, max_tokens)
        return ids, mask

    def __get_lymphnode__(self, patient_id):
        file_path = self.lymph_nodes_paths[patient_id]
        features  = self.read_h5(file_path)
        features  = features.mean(0)
        features  = torch.from_numpy(features).float()
        return features

    def __get_primarytumor__(self, patient_id):

        list_file_paths = self.primary_tumors_paths[patient_id]
        if list_file_paths is not None:
            features = []
            for file_path in self.primary_tumors_paths[patient_id]:
                features.append(self.read_h5(file_path))
            features = np.vstack(features)
        else:
            features = np.zeros((1, 1024))
        features = features.mean(0)
        features  = torch.from_numpy(features).float()

        return features

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, torch.Tensor]]:
        patient_id = self.samples[index]

        # Tabular data
        clinical = self.__get_clinical__(patient_id)
        blood    = self.__get_blood__(patient_id)
        patho    = self.__get_pathological__(patient_id)
        cdm      = self.__get_tma_cdm__(patient_id)

        # Text reports
        h_ids, h_mask  = self.__tokenize__(self.histories[patient_id], self.max_tokens_history)
        s_ids, s_mask  = self.__tokenize__(self.surgeries[patient_id], self.max_tokens_surgery)
        r_ids, r_mask  = self.__tokenize__(self.reports[patient_id],  self.max_tokens_report)
        
        # Lymph node UNI features
        lymph_node = self.__get_lymphnode__(patient_id)

        # Primary tumor UNI features
        primary_tumor = self.__get_primarytumor__(patient_id)

        # Survival status (1 if deceased, 0 otherwise)
        survival_status = self.data_clinical.loc[self.data_clinical.patient_id == patient_id].survival_status.values[0]
        survival_status = torch.tensor(survival_status == "deceased", dtype=torch.int8)
        
        # Days to last information i.e. time to event
        days_to_last_information = self.data_clinical.loc[self.data_clinical.patient_id == patient_id].days_to_last_information.values[0]
        days_to_last_information = torch.tensor(days_to_last_information, dtype=torch.int32)
        
        input = (clinical, blood, patho, cdm, h_ids, h_mask, s_ids, s_mask, r_ids, r_mask, lymph_node, primary_tumor)
        target = (days_to_last_information, survival_status)
        patient_id = torch.tensor(int(patient_id), dtype=torch.int32)

        return patient_id, input, target

    def get_input_dims(self) -> Dict[str, int]:
        """
        Get the actual input dimensions for each modality by examining the first sample.
        
        Returns:
            Dictionary with input dimensions for each tabular modality
        """
        if len(self.samples) == 0:
            raise ValueError("Dataset is empty, cannot determine input dimensions")
        
        # Get first patient to determine dimensions
        first_patient_id = self.samples[0]
        
        # Get actual dimensions from processed data
        clinical_dim     = self.__get_clinical__(first_patient_id).shape[0]
        blood_dim        = self.__get_blood__(first_patient_id).shape[0]
        patho_dim        = self.__get_pathological__(first_patient_id).shape[0]
        cdm_dim          = self.__get_tma_cdm__(first_patient_id).shape[0]
        lymphnode_dim    = self.__get_lymphnode__(first_patient_id).shape[0]
        primarytumor_dim = self.__get_primarytumor__(first_patient_id).shape[0]

        return {
            "clinical": clinical_dim,
            "blood": blood_dim,
            "pathological": patho_dim,
            "cdm": cdm_dim,
            "lymphnode": lymphnode_dim,
            "primarytumor": primarytumor_dim,
        }
    
    def get_tmax(self) -> int:
        """
        Get the maximum time of the dataset.

        Returns:
            Integer value corresponding to the maximum time.
        """
        if len(self.samples) == 0:
            raise ValueError("Dataset is empty, cannot determine input dimensions")
        
        t_max = self.data_clinical.days_to_last_information.max()
        
        t_max = min(t_max, 4963)

        return t_max

    def __str__(self):
        return (
            f"\n--- Dataset ---\n"
            f"  - Split: {self.split}\n"
            f"  - Samples: {len(self.samples)}\n"
            f"  - Max tokens history: {self.max_tokens_history}\n"
            f"  - Max tokens surgery: {self.max_tokens_surgery}\n"
            f"  - Max tokens report: {self.max_tokens_report}\n"
            f"  - Path LM: {self.path_lm}\n"
            f"  - Input dims: {self.get_input_dims()}\n\n"
        )
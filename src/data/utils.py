import torch
import tqdm
import numpy as np
import pandas as pd
from typing import Literal, Tuple
from transformers import AutoModel

from src.data.datamodule.datamodule import HANCOCKDataModule
from src.data.datamodule.h2dg_surv_datamodule import H2DGSurvDataModule


def process_to_array(
    datamodule: HANCOCKDataModule,
    stage: Literal["train", "val", "test", "predict"]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform multimodal HANCOCK data into a flat feature matrix for classical ML models.
    
    Args:
        datamodule: HANCOCK datamodule containing the data loaders
        stage: Data split to process (train/val/test/predict)
    
    Returns:
        Tuple containing:
            - X: Feature matrix (DataFrame) with all modalities concatenated
            - time_to_event: Time until event occurrence
            - survival_status: Censoring status
            - patient_ids: Patient identifiers
    """
    time_to_event, survival_status, patient_ids = [], [], []
    clinica_list, blood_list, patho_list, cdm_list, h_embeddings_list, s_embeddings_list, r_embeddings_list, lymph_list, tumor_list = [], [], [], [], [], [], [], [], []

    # Load text encoder (BioBERT) for text embedding extraction
    text_encoder = AutoModel.from_pretrained(
        datamodule.path_lm, 
        local_files_only=True
    )
    text_encoder.eval()
    
    # Handle HeteroGraph DataModule: use base datasets instead of graph-wrapped ones
    if isinstance(datamodule, H2DGSurvDataModule):
        if stage == "train":
            dataset = datamodule.base_train_dataset
        elif stage == "val":
            dataset = datamodule.base_val_dataset
        elif stage in ["test", "predict"]:
            dataset = datamodule.base_test_dataset
        
        # Create a standard DataLoader for the base dataset
        datadloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=datamodule.batch_size, 
            shuffle=False
        )
    else:
        if stage == "train":
            datadloader = datamodule.train_dataloader()
        elif stage == "val":
            datadloader = datamodule.val_dataloader()
        elif stage in ["test", "predict"]:
            datadloader = datamodule.test_dataloader()
    
    # Process batches and extract text embeddings (CLS token)
    with torch.no_grad():
        for p_id, input, target in tqdm.tqdm(datadloader, desc=f"Parse to Array: {stage} data", unit="batch"):
            time_to_event.append(target[0])
            survival_status.append(target[1])
            patient_ids.append(p_id)
            
            clinical, blood, patho, cdm, h_ids, h_mask, s_ids, s_mask, r_ids, r_mask, lymph, tumor = input
            
            # Squeeze extra dimension from tokenization (shape: [batch, 1, seq] -> [batch, seq])
            h_ids, h_mask = h_ids.squeeze(1), h_mask.squeeze(1)
            s_ids, s_mask = s_ids.squeeze(1), s_mask.squeeze(1)
            r_ids, r_mask = r_ids.squeeze(1), r_mask.squeeze(1)
            
            # Extract text embeddings: histories, surgery descriptions, reports
            h_embeddings = text_encoder(h_ids, h_mask).last_hidden_state[:, 0, :]
            s_embeddings = text_encoder(s_ids, s_mask).last_hidden_state[:, 0, :]
            r_embeddings = text_encoder(r_ids, r_mask).last_hidden_state[:, 0, :]
            
            clinica_list.append(clinical)
            blood_list.append(blood)
            patho_list.append(patho)
            cdm_list.append(cdm)
            h_embeddings_list.append(h_embeddings)
            s_embeddings_list.append(s_embeddings)
            r_embeddings_list.append(r_embeddings)
            lymph_list.append(lymph)
            tumor_list.append(tumor)

    # Concatenate all modalities into arrays
    clinica_array = np.concatenate(clinica_list)
    blood_array = np.concatenate(blood_list)
    patho_array = np.concatenate(patho_list)
    cdm_array = np.concatenate(cdm_list)
    h_embeddings_array = np.concatenate(h_embeddings_list)
    s_embeddings_array = np.concatenate(s_embeddings_list)
    r_embeddings_array = np.concatenate(r_embeddings_list)
    lymph_array = np.concatenate(lymph_list)
    tumor_array = np.concatenate(tumor_list)

    # Early fusion of all features into a single flat matrix
    X = np.concatenate([
        clinica_array,
        blood_array,
        patho_array,
        cdm_array,
        h_embeddings_array,
        s_embeddings_array,
        r_embeddings_array,
        lymph_array,
        tumor_array
    ], axis=1)
    X = pd.DataFrame(X)

    time_to_event = np.concatenate(time_to_event)
    survival_status = np.concatenate(survival_status)
    patient_ids = np.concatenate(patient_ids)
    
    return X, time_to_event, survival_status, patient_ids

#!/usr/bin/env python3
"""
Inference Engine
"""

import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

class InferenceEngine:
    def __init__(self, model_path, scaler_path, encoder_path, hf_tokenizer_path="models/hf_tokenizer", hf_model_path="models/hf_model"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)

        # Load Hugging Face tokenizer and model
        self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)
        self.hf_model = AutoModel.from_pretrained(hf_model_path)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def predict(self, processed_data):
        # Ensure processed_data is a DataFrame
        df = pd.DataFrame([processed_data])

        # Exclude non-feature columns
        numerical_features = [col for col in df.columns if col not in ['datetime', 'reasoning', 'decision', 'signal']]
        
        X_numerical = df[numerical_features]
        # Concatenate multiple reasoning columns into a single string for embedding
        reasoning_columns = ['pattern_recognition', 'context_analysis', 'psychology_assessment', 
                             'execution_decision', 'risk_assessment', 'feature_analysis', 'historical_analysis']
        
        # Ensure all reasoning columns exist, fill missing with empty string
        for col in reasoning_columns:
            if col not in df.columns:
                df[col] = ''

        X_reasoning = df[reasoning_columns].agg(' '.join, axis=1).fillna('')

        # Scale numerical features
        X_numerical_scaled = self.scaler.transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features, index=df.index)

        # Generate text embeddings using the loaded Hugging Face model
        encoded_input = self.hf_tokenizer(X_reasoning.tolist(), padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.hf_model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        X_reasoning_embeddings_df = pd.DataFrame(sentence_embeddings.numpy(), index=df.index)
        X_reasoning_embeddings_df.columns = [f'reasoning_embed_{i}' for i in range(X_reasoning_embeddings_df.shape[1])]

        # Combine features
        X_processed = pd.concat([X_numerical_scaled_df, X_reasoning_embeddings_df], axis=1)

        # Make prediction
        predictions_encoded = self.model.predict(X_processed)
        
        # Decode prediction
        prediction = self.label_encoder.inverse_transform(predictions_encoded)
        
        return prediction[0]

import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import joblib

class EmbeddingGenerator:
    def __init__(self, hf_tokenizer_path="models/hf_tokenizer", 
                 hf_model_path="models/hf_model",
                 embeddings_output_dir="data/processed/embeddings"):
        
        self.hf_tokenizer_path = hf_tokenizer_path
        self.hf_model_path = hf_model_path
        self.embeddings_output_dir = embeddings_output_dir
        
        os.makedirs(self.embeddings_output_dir, exist_ok=True)

        # Load Hugging Face tokenizer and model (ensure they are downloaded/cached)
        try:
            print("Attempting to load Hugging Face model and tokenizer from local storage...")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path)
            self.hf_model = AutoModel.from_pretrained(self.hf_model_path)
            print("Successfully loaded Hugging Face model and tokenizer from local storage.")
        except Exception as e:
            print(f"Failed to load from local storage: {e}. Downloading Hugging Face model and tokenizer...")
            self.hf_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.hf_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.hf_tokenizer.save_pretrained(self.hf_tokenizer_path)
            self.hf_model.save_pretrained(self.hf_model_path)
            print("Successfully downloaded and saved Hugging Face model and tokenizer.")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_and_save_embeddings(self, input_file_path, batch_size=32):
        """Generates embeddings for the reasoning column and saves them to a file."""
        print(f"Processing file: {input_file_path}")
        df = pd.read_csv(input_file_path)
        X_reasoning = df['reasoning'].fillna('').tolist()

        all_embeddings = []
        for i in range(0, len(X_reasoning), batch_size):
            batch = X_reasoning[i:i+batch_size]
            encoded_input = self.hf_tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.hf_model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            all_embeddings.append(sentence_embeddings.numpy())
        
        embeddings_array = torch.cat([torch.from_numpy(e) for e in all_embeddings]).numpy()

        # Save embeddings
        output_filename = f"embeddings_{Path(input_file_path).stem.replace('final_', '')}.joblib"
        output_path = Path(self.embeddings_output_dir) / output_filename
        joblib.dump(embeddings_array, output_path)
        print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    
    # Example usage: Process all final data files
    final_data_dir = "data/final"
    for file_name in os.listdir(final_data_dir):
        if file_name.startswith("final_") and file_name.endswith(".csv"):
            file_path = os.path.join(final_data_dir, file_name)
            generator.generate_and_save_embeddings(file_path)

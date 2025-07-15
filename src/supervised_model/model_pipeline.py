import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path

class SupervisedModelPipeline:
    def __init__(self, model_save_path="models/supervised_model.joblib", 
                 scaler_save_path="models/scaler.joblib", 
                 label_encoder_save_path="models/label_encoder.joblib", 
                 hf_tokenizer_path="models/hf_tokenizer", 
                 hf_model_path="models/hf_model"):
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.hf_tokenizer = None
        self.hf_model = None
        
        self.model_save_path = model_save_path
        self.scaler_save_path = scaler_save_path
        self.label_encoder_save_path = label_encoder_save_path
        self.hf_tokenizer_path = hf_tokenizer_path
        self.hf_model_path = hf_model_path
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        os.makedirs(hf_tokenizer_path, exist_ok=True)
        os.makedirs(hf_model_path, exist_ok=True)

        # Try loading Hugging Face tokenizer and model from local storage first
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

    def load_data(self, file_path):
        """Loads data from a CSV file."""
        df = pd.read_csv(file_path)
        # Ensure datetime is parsed correctly
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def prepare_data(self, df, embeddings_file_path):
        """Prepares data for model training, including scaling and loading pre-computed text embeddings."""
        # Exclude 'signal' column and 'datetime' from features
        numerical_features = [col for col in df.columns if col not in ['datetime', 'reasoning', 'decision', 'signal']]
        
        X_numerical = df[numerical_features]
        y = df['decision']

        # Scale numerical features
        self.scaler = StandardScaler()
        X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features, index=df.index)

        # Load pre-computed text embeddings
        embeddings_array = joblib.load(embeddings_file_path)
        X_reasoning_embeddings_df = pd.DataFrame(embeddings_array, index=df.index)
        X_reasoning_embeddings_df.columns = [f'reasoning_embed_{i}' for i in range(X_reasoning_embeddings_df.shape[1])]

        # Combine numerical and text features
        X = pd.concat([X_numerical_scaled_df, X_reasoning_embeddings_df], axis=1)

        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded

    def train_model(self, X_train, y_train):
        """Trains the RandomForestClassifier model."""
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """Evaluates the trained model."""
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, labels=self.label_encoder.transform(self.label_encoder.classes_)))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        return accuracy_score(y_test, y_pred)

    def save_model(self):
        """Saves the trained model, scaler, label encoder, and Hugging Face components."""
        joblib.dump(self.model, self.model_save_path)
        joblib.dump(self.scaler, self.scaler_save_path)
        joblib.dump(self.label_encoder, self.label_encoder_save_path)
        # Hugging Face model and tokenizer are saved within prepare_data if downloaded
        print(f"Supervised model components saved to {os.path.dirname(self.model_save_path)} and Hugging Face components to {self.hf_model_path}")

    def load_saved_model(self):
        """Loads the trained model, scaler, label encoder, and Hugging Face components."""
        self.model = joblib.load(self.model_save_path)
        self.scaler = joblib.load(self.scaler_save_path)
        self.label_encoder = joblib.load(self.label_encoder_save_path)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path)
        self.hf_model = AutoModel.from_pretrained(self.hf_model_path)
        print(f"Supervised model components loaded from {os.path.dirname(self.model_save_path)} and Hugging Face components from {self.hf_model_path}")

    def predict(self, new_data_df):
        """Makes predictions using the loaded model."""
        # Ensure new_data_df has the same structure as training data
        numerical_features = [col for col in new_data_df.columns if col not in ['datetime', 'reasoning', 'decision', 'signal']]
        
        X_numerical = new_data_df[numerical_features]
        X_reasoning = new_data_df['reasoning'].fillna('')

        # Scale numerical features using the fitted scaler
        X_numerical_scaled = self.scaler.transform(X_numerical)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features, index=new_data_df.index)

        # Generate text embeddings using the loaded Hugging Face model
        encoded_input = self.hf_tokenizer(X_reasoning.tolist(), padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.hf_model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        X_reasoning_embeddings_df = pd.DataFrame(sentence_embeddings.numpy(), index=new_data_df.index)
        X_reasoning_embeddings_df.columns = [f'reasoning_embed_{i}' for i in range(X_reasoning_embeddings_df.shape[1])]

        # Combine features
        X_processed = pd.concat([X_numerical_scaled_df, X_reasoning_embeddings_df], axis=1)

        # Make prediction
        predictions_encoded = self.model.predict(X_processed)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions

if __name__ == "__main__":
    # Example Usage:
    pipeline = SupervisedModelPipeline()

    # Load data (using one of the final files for demonstration)
    data_file = "data/final/final_Nifty_2.csv"
    df = pipeline.load_data(data_file)

    # Prepare data
    # Construct embeddings file path
    embeddings_file = Path("data/processed/embeddings") / f"embeddings_{Path(data_file).stem.replace('final_', '')}.joblib"
    X, y = pipeline.prepare_data(df, embeddings_file)

    # Time-series split
    # Sort by datetime to ensure chronological order
    df_sorted = df.sort_values(by='datetime').reset_index(drop=True)
    
    # Get the corresponding X and y after sorting
    X_sorted = X.loc[df_sorted.index]
    y_sorted = y[df_sorted.index]

    split_point = int(len(df_sorted) * 0.8)
    X_train, X_test = X_sorted.iloc[:split_point], X_sorted.iloc[split_point:]
    y_train, y_test = y_sorted[:split_point], y_sorted[split_point:]

    print(f"Data split: Training samples = {len(X_train)}, Test samples = {len(X_test)}")
    print(f"y_train class distribution:\n{pd.Series(y_train).value_counts()}")
    print(f"y_test class distribution:\n{pd.Series(y_test).value_counts()}")

    # Train model
    pipeline.train_model(X_train, y_train)

    # Evaluate model
    pipeline.evaluate_model(X_test, y_test)

    # Save model components
    pipeline.save_model()

    # Example of loading and predicting
    # new_pipeline = SupervisedModelPipeline()
    # new_pipeline.load_saved_model()
    # predictions = new_pipeline.predict(df.head()) # Predict on first few rows of original data
    # print("Predictions on sample data:", predictions)
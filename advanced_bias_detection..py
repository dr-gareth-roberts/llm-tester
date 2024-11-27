import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class AdvancedBiasDetector:
    def __init__(self):
        # Load advanced NLP models
        self.nlp = spacy.load('en_core_web_trf')  # Transformer-based model
        
        # Bias detection transformer model
        self.bias_model = AutoModelForSequenceClassification.from_pretrained(
            'facebook/roberta-hate-speech-dynabench-r4-target'
        )
        self.bias_tokenizer = AutoTokenizer.from_pretrained(
            'facebook/roberta-hate-speech-dynabench-r4-target'
        )
        
        # Embedding model for semantic analysis
        self.embedding_model = AutoModelForSequenceClassification.from_pretrained(
            'sentence-transformers/all-mpnet-base-v2'
        )
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/all-mpnet-base-v2'
        )
    
    def detect_intersectional_bias(self, text: str) -> Dict[str, float]:
        """
        Detect nuanced, intersectional biases
        
        :param text: Input text
        :return: Bias scores across multiple dimensions
        """
        # Demographic intersectionality analysis
        intersectional_dimensions = [
            'race', 'gender', 'age', 'socioeconomic_status', 
            'disability', 'sexual_orientation'
        ]
        
        # Prepare embedding
        inputs = self.embedding_tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            embeddings = self.embedding_model(**inputs).last_hidden_state.mean(dim=1)
        
        # Bias detection using transformer model
        bias_inputs = self.bias_tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            bias_output = torch.softmax(self.bias_model(**bias_inputs).logits, dim=1)
        
        # Comprehensive bias analysis
        bias_scores = {
            dim: self._calculate_dimensional_bias(embeddings, dim)
            for dim in intersectional_dimensions
        }
        
        # Add hate speech probability
        bias_scores['hate_speech_prob'] = bias_output[0][1].item()
        
        return bias_scores
    
    def _calculate_dimensional_bias(
        self, 
        text_embedding: torch.Tensor, 
        dimension: str
    ) -> float:
        """
        Calculate bias for a specific demographic dimension
        
        :param text_embedding: Text embedding
        :param dimension: Bias dimension
        :return: Bias score
        """
        # Curated reference embeddings for various biased contexts
        biased_references = {
            'race': [
                "stereotypical statements about racial groups",
                "racially charged language",
                "discriminatory racial narratives"
            ],
            'gender': [
                "sexist language",
                "gender role stereotypes",
                "discriminatory gender narratives"
            ],
            # Add more reference sets for other dimensions
        }
        
        # Calculate semantic similarity with biased references
        reference_embeddings = self._get_reference_embeddings(
            biased_references.get(dimension, [])
        )
        
        # Cosine similarity calculation
        similarities = cosine_similarity(
            text_embedding.numpy(), 
            reference_embeddings
        )
        
        return float(np.mean(similarities))
    
    def _get_reference_embeddings(self, references: List[str]) -> np.ndarray:
        """
        Generate embeddings for reference texts
        
        :param references: List of reference texts
        :return: Reference embeddings
        """
        embeddings = []
        for ref in references:
            inputs = self.embedding_tokenizer(
                ref, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                embedding = self.embedding_model(**inputs).last_hidden_state.mean(dim=1)
            
            embeddings.append(embedding.numpy())
        
        return np.vstack(embeddings)
    
    def generate_bias_report(self, texts: List[str]) -> pd.DataFrame:
        """
        Generate comprehensive bias report for multiple texts
        
        :param texts: List of texts to analyze
        :return: DataFrame with bias analysis
        """
        bias_results = []
        
        for text in texts:
            bias_scores = self.detect_intersectional_bias(text)
            bias_results.append({
                'text': text,
                **bias_scores
            })
        
        return pd.DataFrame(bias_results)

# Example usage
def main():
    # Initialize advanced bias detector
    bias_detector = AdvancedBiasDetector()
    
    # Sample texts for bias analysis
    sample_texts = [
        "Women are naturally better at caregiving jobs.",
        "Men are more suitable for leadership positions.",
        "People from certain ethnic backgrounds are more intelligent.",
        "Disabled individuals cannot be as productive in the workplace."
    ]
    
    # Generate comprehensive bias report
    bias_report = bias_detector.generate_bias_report(sample_texts)
    
    # Display results
    print(bias_report)
    
    # Optional: Visualize bias scores
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        bias_report.drop(columns=['text']), 
        annot=True, 
        cmap='YlOrRd'
    )
    plt.title('Intersectional Bias Analysis')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
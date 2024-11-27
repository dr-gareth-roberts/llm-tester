import numpy as np
import textstat
from typing import Dict, Any
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import spacy

class AdvancedMetricsCalculator:
    def __init__(self):
        # Load spaCy model for advanced NLP metrics
        self.nlp = spacy.load('en_core_web_sm')
        self.rouge = Rouge()

    def calculate_comprehensive_metrics(self, response: str, prompt: str) -> Dict[str, Any]:
        metrics = {}

        # Text complexity metrics
        metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(response)
        metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(response)
        metrics['smog_index'] = textstat.smog_index(response)

        # Linguistic complexity
        doc = self.nlp(response)
        metrics['noun_chunks'] = len(list(doc.noun_chunks))
        metrics['named_entities'] = len(list(doc.ents))
        
        # Linguistic diversity
        unique_words = len(set(token.text.lower() for token in doc if not token.is_stop and not token.is_punct))
        total_words = len([token for token in doc if not token.is_stop and not token.is_punct])
        metrics['lexical_diversity'] = unique_words / total_words if total_words > 0 else 0

        # Semantic similarity (basic)
        prompt_doc = self.nlp(prompt)
        response_doc = self.nlp(response)
        metrics['prompt_response_similarity'] = prompt_doc.similarity(response_doc)

        # Translation quality metrics
        try:
            rouge_scores = self.rouge.get_scores(response, prompt)
            metrics['rouge_1'] = rouge_scores[0]['rouge-1']['f']
            metrics['rouge_2'] = rouge_scores[0]['rouge-2']['f']
            metrics['rouge_l'] = rouge_scores[0]['rouge-l']['f']
        except Exception:
            metrics['rouge_1'] = metrics['rouge_2'] = metrics['rouge_l'] = 0

        # Language model specific metrics
        metrics['perplexity'] = self._calculate_perplexity(response)

        return metrics

    def _calculate_perplexity(self, text: str, base: int = 2) -> float:
        """
        Calculate perplexity of the text (simplified implementation)
        """
        doc = self.nlp(text)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        # Simplified log probability calculation
        log_prob = np.mean([np.log2(1/len(tokens)) for _ in tokens])
        perplexity = np.power(base, -log_prob)
        
        return perplexity
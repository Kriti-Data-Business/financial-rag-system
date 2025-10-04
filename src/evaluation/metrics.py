"""
Evaluation metrics for RAG system performance.
"""
from typing import List, Dict
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGMetrics:
    """Calculate evaluation metrics for RAG system."""
    
    @staticmethod
    def calculate_retrieval_metrics(
        retrieved_docs: List[Dict],
        relevant_docs: List[str]
    ) -> Dict[str, float]:
        """
        Calculate retrieval metrics (precision, recall, F1).
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            relevant_docs: List of relevant document IDs
            
        Returns:
            Dictionary of metrics
        """
        retrieved_ids = [doc['id'] for doc in retrieved_docs]
        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_ids)
        
        # True positives
        tp = len(relevant_set & retrieved_set)
        
        # False positives
        fp = len(retrieved_set - relevant_set)
        
        # False negatives
        fn = len(relevant_set - retrieved_set)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    @staticmethod
    def calculate_mrr(retrieved_docs: List[Dict], relevant_doc_id: str) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved_docs: List of retrieved documents (ordered by relevance)
            relevant_doc_id: ID of the relevant document
            
        Returns:
            MRR score
        """
        for i, doc in enumerate(retrieved_docs, 1):
            if doc['id'] == relevant_doc_id:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def calculate_ndcg(retrieved_docs: List[Dict], relevance_scores: Dict[str, float], k: int = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevance_scores: Dictionary mapping doc IDs to relevance scores
            k: Number of top results to consider
            
        Returns:
            NDCG score
        """
        if k:
            retrieved_docs = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs, 1):
            relevance = relevance_scores.get(doc['id'], 0)
            dcg += relevance / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)
        if k:
            ideal_scores = ideal_scores[:k]
        
        idcg = 0.0
        for i, score in enumerate(ideal_scores, 1):
            idcg += score / np.log2(i + 1)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    @staticmethod
    def calculate_answer_similarity(predicted: str, reference: str) -> float:
        """
        Calculate simple token-based similarity between answers.
        
        Args:
            predicted: Predicted answer
            reference: Reference answer
            
        Returns:
            Similarity score (0-1)
        """
        # Simple token overlap
        predicted_tokens = set(predicted.lower().split())
        reference_tokens = set(reference.lower().split())
        
        intersection = predicted_tokens & reference_tokens
        union = predicted_tokens | reference_tokens
        
        return len(intersection) / len(union) if union else 0.0
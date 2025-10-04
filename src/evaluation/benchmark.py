"""
Benchmarking tools for RAG system evaluation.
"""
import time
from typing import List, Dict
import json
from pathlib import Path
from src.data.database.vector_store import VectorStore
from src.models.llm import LLMModel
from src.evaluation.metrics import RAGMetrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGBenchmark:
    """Benchmark the RAG system performance."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.vector_store = VectorStore()
        self.llm_model = LLMModel()
        self.metrics = RAGMetrics()
        
    def run_benchmark(
        self,
        test_queries: List[Dict],
        output_file: str = None
    ) -> Dict:
        """
        Run benchmark on test queries.
        
        Args:
            test_queries: List of test query dictionaries with:
                - query: The question
                - relevant_docs: List of relevant document IDs
                - expected_answer: Optional expected answer
            output_file: Optional file to save results
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Running benchmark on {len(test_queries)} queries")
        
        results = {
            'total_queries': len(test_queries),
            'query_results': [],
            'aggregate_metrics': {}
        }
        
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_mrrs = []
        total_time = 0
        
        for i, test_query in enumerate(test_queries, 1):
            logger.info(f"Processing query {i}/{len(test_queries)}")
            
            query = test_query['query']
            relevant_docs = test_query.get('relevant_docs', [])
            
            # Measure retrieval time
            start_time = time.time()
            
            # Retrieve documents
            retrieved_docs = self.vector_store.search(query=query, top_k=5)
            
            retrieval_time = time.time() - start_time
            
            # Calculate retrieval metrics
            if relevant_docs:
                retrieval_metrics = self.metrics.calculate_retrieval_metrics(
                    retrieved_docs, relevant_docs
                )
                mrr = self.metrics.calculate_mrr(retrieved_docs, relevant_docs[0])
            else:
                retrieval_metrics = {}
                mrr = 0.0
            
            # Generate answer
            if retrieved_docs:
                context = "\n\n".join([doc['content'] for doc in retrieved_docs])
                start_time = time.time()
                answer = self.llm_model.generate_response(query, context)
                generation_time = time.time() - start_time
            else:
                answer = "No relevant documents found"
                generation_time = 0
            
            # Calculate answer similarity if expected answer provided
            answer_similarity = None
            if 'expected_answer' in test_query:
                answer_similarity = self.metrics.calculate_answer_similarity(
                    answer, test_query['expected_answer']
                )
            
            # Store results
            query_result = {
                'query': query,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': retrieval_time + generation_time,
                'retrieved_docs_count': len(retrieved_docs),
                'retrieval_metrics': retrieval_metrics,
                'mrr': mrr,
                'answer': answer,
                'answer_similarity': answer_similarity
            }
            
            results['query_results'].append(query_result)
            
            # Aggregate metrics
            if retrieval_metrics:
                all_precisions.append(retrieval_metrics['precision'])
                all_recalls.append(retrieval_metrics['recall'])
                all_f1s.append(retrieval_metrics['f1_score'])
            all_mrrs.append(mrr)
            total_time += retrieval_time + generation_time
        
        # Calculate aggregate metrics
        results['aggregate_metrics'] = {
            'avg_precision': sum(all_precisions) / len(all_precisions) if all_precisions else 0,
            'avg_recall': sum(all_recalls) / len(all_recalls) if all_recalls else 0,
            'avg_f1': sum(all_f1s) / len(all_f1s) if all_f1s else 0,
            'mean_mrr': sum(all_mrrs) / len(all_mrrs) if all_mrrs else 0,
            'avg_query_time': total_time / len(test_queries),
            'total_time': total_time
        }
        
        logger.info("Benchmark completed")
        logger.info(f"Aggregate Metrics: {results['aggregate_metrics']}")
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a human-readable report from benchmark results.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("RAG SYSTEM BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append(f"Total Queries: {results['total_queries']}")
        report.append("")
        
        # Aggregate metrics
        metrics = results['aggregate_metrics']
        report.append("AGGREGATE METRICS:")
        report.append("-" * 80)
        report.append(f"Average Precision: {metrics['avg_precision']:.4f}")
        report.append(f"Average Recall:    {metrics['avg_recall']:.4f}")
        report.append(f"Average F1 Score:  {metrics['avg_f1']:.4f}")
        report.append(f"Mean MRR:          {metrics['mean_mrr']:.4f}")
        report.append(f"Avg Query Time:    {metrics['avg_query_time']:.4f}s")
        report.append(f"Total Time:        {metrics['total_time']:.4f}s")
        report.append("")# Per-query details
        report.append("PER-QUERY RESULTS:")
        report.append("-" * 80)
        
        for i, query_result in enumerate(results['query_results'], 1):
            report.append(f"\nQuery {i}: {query_result['query']}")
            report.append(f"  Retrieval Time: {query_result['retrieval_time']:.4f}s")
            report.append(f"  Generation Time: {query_result['generation_time']:.4f}s")
            report.append(f"  Retrieved Docs: {query_result['retrieved_docs_count']}")
            
            if query_result['retrieval_metrics']:
                rm = query_result['retrieval_metrics']
                report.append(f"  Precision: {rm['precision']:.4f}")
                report.append(f"  Recall: {rm['recall']:.4f}")
                report.append(f"  F1 Score: {rm['f1_score']:.4f}")
            
            report.append(f"  MRR: {query_result['mrr']:.4f}")
            
            if query_result['answer_similarity'] is not None:
                report.append(f"  Answer Similarity: {query_result['answer_similarity']:.4f}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    benchmark = RAGBenchmark()
    
    # Sample test queries
    test_queries = [
        {
            "query": "What is machine learning?",
            "relevant_docs": ["doc1", "doc2"],
            "expected_answer": "Machine learning is a subset of artificial intelligence..."
        },
        # Add more test queries
    ]
    
    # Run benchmark
    results = benchmark.run_benchmark(test_queries, output_file="benchmark_results.json")
    
    # Generate and print report
    report = benchmark.generate_report(results)
    print(report)
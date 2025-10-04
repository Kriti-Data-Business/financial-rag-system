"""
Testing utilities for the RAG system.
"""
from typing import List, Dict
from src.data.database.vector_store import VectorStore
from src.models.llm import LLMModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGTester:
    """Test the RAG system with various scenarios."""
    
    def __init__(self):
        """Initialize tester."""
        self.vector_store = VectorStore()
        self.llm_model = LLMModel()
    
    def test_retrieval(self, query: str, expected_doc_ids: List[str] = None) -> Dict:
        """
        Test document retrieval.
        
        Args:
            query: Test query
            expected_doc_ids: Optional list of expected document IDs
            
        Returns:
            Test results
        """
        logger.info(f"Testing retrieval for: {query}")
        
        results = self.vector_store.search(query, top_k=5)
        
        retrieved_ids = [r['id'] for r in results]
        
        test_result = {
            'query': query,
            'retrieved_count': len(results),
            'retrieved_ids': retrieved_ids,
            'results': results
        }
        
        if expected_doc_ids:
            matches = [doc_id for doc_id in retrieved_ids if doc_id in expected_doc_ids]
            test_result['expected_matches'] = len(matches)
            test_result['match_rate'] = len(matches) / len(expected_doc_ids)
        
        return test_result
    
    def test_generation(self, query: str, context: str) -> Dict:
        """
        Test answer generation.
        
        Args:
            query: Test query
            context: Context for generation
            
        Returns:
            Test results
        """
        logger.info(f"Testing generation for: {query}")
        
        answer = self.llm_model.generate_response(query, context)
        
        return {
            'query': query,
            'context_length': len(context),
            'answer': answer,
            'answer_length': len(answer)
        }
    
    def test_end_to_end(self, query: str) -> Dict:
        """
        Test complete RAG pipeline.
        
        Args:
            query: Test query
            
        Returns:
            Test results
        """
        logger.info(f"Testing end-to-end for: {query}")
        
        # Retrieval
        results = self.vector_store.search(query, top_k=5)
        
        if not results:
            return {
                'query': query,
                'status': 'no_documents_found',
                'answer': None
            }
        
        # Prepare context
        context = "\n\n".join([r['content'] for r in results])
        
        # Generation
        answer = self.llm_model.generate_response(query, context)
        
        return {
            'query': query,
            'status': 'success',
            'retrieved_docs': len(results),
            'context_length': len(context),
            'answer': answer,
            'sources': [{'id': r['id'], 'score': r['score']} for r in results]
        }
    
    def run_test_suite(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Run a suite of test cases.
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            List of test results
        """
        logger.info(f"Running test suite with {len(test_cases)} cases")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Running test case {i}/{len(test_cases)}")
            
            test_type = test_case.get('type', 'end_to_end')
            
            if test_type == 'retrieval':
                result = self.test_retrieval(
                    test_case['query'],
                    test_case.get('expected_doc_ids')
                )
            elif test_type == 'generation':
                result = self.test_generation(
                    test_case['query'],
                    test_case['context']
                )
            else:
                result = self.test_end_to_end(test_case['query'])
            
            result['test_id'] = i
            result['test_type'] = test_type
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage
    tester = RAGTester()
    
    # Test retrieval
    retrieval_result = tester.test_retrieval("What is Python?")
    print("Retrieval Result:", retrieval_result)
    
    # Test end-to-end
    e2e_result = tester.test_end_to_end("Explain machine learning")
    print("End-to-End Result:", e2e_result)
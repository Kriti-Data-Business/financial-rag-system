# Main RAG implementation
# src/models/rag_system.py
# Main RAG implementation for Australian Financial Investment Planning

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
import json

from ..data.database.chroma_manager import ChromaDBManager
from ..data.processors.document_processor import DocumentProcessor
from .embedding_manager import EmbeddingManager
from .financial_calculator import FinancialCalculator
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class AustralianFinancialRAGSystem:
    """
    Main RAG system for Australian financial investment planning advice.
    Integrates retrieval, generation, and financial calculations.
    """
    
    def __init__(self, config_path: str = "config/development.yaml"):
        """Initialize the RAG system with configuration."""
        self.config = Config(config_path)
        self.embedding_manager = EmbeddingManager(self.config)
        self.chroma_manager = ChromaDBManager(self.config)
        self.document_processor = DocumentProcessor(self.config)
        self.financial_calculator = FinancialCalculator()
        
        logger.info("Australian Financial RAG System initialized")
    
    def initialize_knowledge_base(self, data_path: str = "data/processed/documents") -> None:
        """Initialize the knowledge base with processed documents."""
        try:
            documents = self.document_processor.load_processed_documents(data_path)
            self.chroma_manager.add_documents(documents)
            logger.info(f"Knowledge base initialized with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise
    
    def get_financial_advice(self, 
                           query: str, 
                           user_profile: Optional[Dict] = None,
                           num_results: int = 5) -> Dict[str, Any]:
        """
        Get comprehensive financial advice for Australian investors.
        
        Args:
            query: User's financial question
            user_profile: User's financial information (salary, expenses, age, etc.)
            num_results: Number of documents to retrieve
            
        Returns:
            Dictionary containing advice, calculations, and context
        """
        try:
            # Enhance query for better retrieval
            enhanced_query = self._enhance_query(query)
            
            # Retrieve relevant documents
            search_results = self.chroma_manager.search(
                query=enhanced_query,
                n_results=num_results
            )
            
            # Apply financial calculations if user profile provided
            calculations = {}
            if user_profile:
                calculations = self._apply_financial_calculations(user_profile)
            
            # Generate response
            response = self._generate_response(
                query=query,
                context=search_results.get('documents', []),
                calculations=calculations,
                user_profile=user_profile
            )
            
            return {
                'response': response,
                'context': search_results.get('documents', []),
                'metadata': search_results.get('metadatas', []),
                'financial_calculations': calculations,
                'enhanced_query': enhanced_query,
                'sources': self._extract_sources(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error generating financial advice: {e}")
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again.",
                'error': str(e)
            }
    
    def _enhance_query(self, query: str) -> str:
        """Enhance user query with Australian financial context."""
        enhanced = query.lower()
        
        # Australian financial term expansions
        term_expansions = {
            'super': 'superannuation retirement savings',
            'shares': 'equities stocks ASX',
            'property': 'real estate REIT investment',
            'savings': 'cash deposit term deposit high yield',
            'tax': 'taxation CGT capital gains franking credits',
            'retirement': 'superannuation pension age pension',
            'emergency fund': 'emergency savings buffer cash reserve',
            'investment': 'ASX ETF managed funds portfolio',
            'gold': 'gold bullion precious metals AUD',
            'silver': 'silver precious metals commodity AUD'
        }
        
        for term, expansion in term_expansions.items():
            if term in enhanced:
                enhanced = enhanced.replace(term, f"{term} {expansion}")
        
        # Add Australian context
        if any(word in enhanced for word in ['investment', 'portfolio', 'allocation']):
            enhanced += " Australian market ASX AUD"
        
        if any(word in enhanced for word in ['tax', 'taxation']):
            enhanced += " Australian taxation system ATO"
            
        return enhanced
    
    def _apply_financial_calculations(self, user_profile: Dict) -> Dict[str, Any]:
        """Apply Australian financial calculations based on user profile."""
        calculations = {}
        
        try:
            # Emergency fund calculation
            if 'monthly_expenses' in user_profile:
                calculations['emergency_fund'] = self.financial_calculator.calculate_emergency_fund(
                    user_profile['monthly_expenses']
                )
            
            # Superannuation calculations
            if 'annual_salary' in user_profile:
                calculations['super_guarantee'] = self.financial_calculator.calculate_super_guarantee(
                    user_profile['annual_salary']
                )
                calculations['salary_sacrifice_benefit'] = self.financial_calculator.calculate_salary_sacrifice_benefit(
                    user_profile['annual_salary'],
                    user_profile.get('marginal_tax_rate', 0.325)
                )
            
            # Investment allocation
            if 'annual_salary' in user_profile and 'monthly_expenses' in user_profile:
                disposable_income = user_profile['annual_salary'] - (user_profile['monthly_expenses'] * 12)
                if disposable_income > 0:
                    calculations['investment_allocation'] = self.financial_calculator.calculate_investment_allocation(
                        disposable_income,
                        user_profile.get('age', 35)
                    )
            
            # Risk assessment
            if all(key in user_profile for key in ['age', 'annual_salary', 'monthly_expenses']):
                calculations['risk_profile'] = self.financial_calculator.assess_risk_profile(
                    user_profile['age'],
                    user_profile['annual_salary'],
                    user_profile['monthly_expenses'] * 12
                )
            
        except Exception as e:
            logger.error(f"Error in financial calculations: {e}")
            calculations['error'] = "Unable to complete financial calculations"
        
        return calculations
    
    def _generate_response(self, 
                          query: str, 
                          context: List[str], 
                          calculations: Dict, 
                          user_profile: Optional[Dict] = None) -> str:
        """Generate financial advice response."""
        
        # Create structured prompt
        prompt_parts = [
            "You are a knowledgeable Australian financial advisor providing practical investment advice.",
            "Base your response on the following information and calculations.\n"
        ]
        
        # Add user context
        if user_profile:
            prompt_parts.append(f"User Profile:")
            if 'age' in user_profile:
                prompt_parts.append(f"- Age: {user_profile['age']}")
            if 'annual_salary' in user_profile:
                prompt_parts.append(f"- Annual Salary: ${user_profile['annual_salary']:,} AUD")
            if 'monthly_expenses' in user_profile:
                prompt_parts.append(f"- Monthly Expenses: ${user_profile['monthly_expenses']:,} AUD")
            prompt_parts.append("")
        
        # Add calculations
        if calculations:
            prompt_parts.append("Financial Calculations:")
            for key, value in calculations.items():
                if isinstance(value, dict):
                    prompt_parts.append(f"- {key.replace('_', ' ').title()}:")
                    for subkey, subvalue in value.items():
                        prompt_parts.append(f"  * {subkey.replace('_', ' ').title()}: {subvalue}")
                else:
                    prompt_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
            prompt_parts.append("")
        
        # Add context information
        if context:
            prompt_parts.append("Relevant Financial Information:")
            for i, ctx in enumerate(context[:3], 1):
                prompt_parts.append(f"{i}. {ctx[:300]}...")
            prompt_parts.append("")
        
        prompt_parts.extend([
            f"User Question: {query}",
            "",
            "Provide clear, practical advice for Australian investors. Include specific recommendations",
            "and explain the reasoning. Mention relevant Australian regulations, tax implications,",
            "and investment options where appropriate.",
            "",
            "Response:"
        ])
        
        # For now, return a structured response based on available information
        # In a full implementation, this would use a local LLM
        return self._create_structured_response(query, context, calculations, user_profile)
    
    def _create_structured_response(self, query: str, context: List[str], 
                                  calculations: Dict, user_profile: Optional[Dict]) -> str:
        """Create a structured response based on available information."""
        
        response_parts = []
        
        # Determine response type based on query
        query_lower = query.lower()
        
        if 'emergency fund' in query_lower:
            response_parts.extend(self._emergency_fund_advice(calculations, user_profile))
        elif 'super' in query_lower or 'retirement' in query_lower:
            response_parts.extend(self._superannuation_advice(calculations, user_profile))
        elif 'investment' in query_lower or 'portfolio' in query_lower:
            response_parts.extend(self._investment_advice(calculations, user_profile, context))
        elif 'gold' in query_lower or 'silver' in query_lower or 'precious metals' in query_lower:
            response_parts.extend(self._precious_metals_advice(context))
        else:
            response_parts.extend(self._general_financial_advice(query, context, calculations))
        
        # Add disclaimer
        response_parts.append("\n⚠️ This advice is general in nature and doesn't consider your specific circumstances. Consider seeking professional financial advice before making investment decisions.")
        
        return "\n".join(response_parts)
    
    def _emergency_fund_advice(self, calculations: Dict, user_profile: Optional[Dict]) -> List[str]:
        """Generate emergency fund specific advice."""
        advice = ["💰 Emergency Fund Guidance:"]
        
        if 'emergency_fund' in calculations:
            advice.append(f"Based on your expenses, you should aim for ${calculations['emergency_fund']:,.0f} in emergency savings.")
            advice.append("This covers 6 months of essential expenses as recommended for Australian households.")
        else:
            advice.append("A general rule is to save 3-6 months of essential expenses in an easily accessible account.")
        
        advice.extend([
            "",
            "Best options for emergency funds in Australia:",
            "• High-yield online savings accounts (currently 4.5-5.5% p.a.)",
            "• Term deposits for portion of funds (guaranteed returns)",
            "• Avoid investment accounts due to volatility risk",
            "",
            "Keep your emergency fund separate from everyday banking and investment accounts."
        ])
        
        return advice
    
    def _superannuation_advice(self, calculations: Dict, user_profile: Optional[Dict]) -> List[str]:
        """Generate superannuation specific advice."""
        advice = ["🏛️ Superannuation Strategy:"]
        
        if 'super_guarantee' in calculations:
            advice.append(f"Your employer contributes ${calculations['super_guarantee']:,.0f} annually (11% super guarantee).")
        
        if 'salary_sacrifice_benefit' in calculations:
            benefit = calculations['salary_sacrifice_benefit']
            advice.append(f"Salary sacrificing could save you ${benefit.get('annual_tax_saving', 0):,.0f} in taxes annually.")
        
        advice.extend([
            "",
            "Key superannuation strategies:",
            "• Maximize employer contributions before considering additional contributions",
            "• Salary sacrifice is most beneficial for those in higher tax brackets (32.5%+)",
            "• Consider government co-contribution if eligible (income under $58,445)",
            "• Review investment options - balanced funds vs growth vs conservative",
            "",
            f"Annual contribution limits: Concessional $27,500, Non-concessional $110,000"
        ])
        
        return advice
    
    def _investment_advice(self, calculations: Dict, user_profile: Optional[Dict], context: List[str]) -> List[str]:
        """Generate investment specific advice."""
        advice = ["📈 Investment Strategy for Australian Investors:"]
        
        if 'investment_allocation' in calculations:
            allocation = calculations['investment_allocation']
            advice.append(f"Suggested allocation based on your profile:")
            for asset_class, percentage in allocation.items():
                advice.append(f"• {asset_class.replace('_', ' ').title()}: {percentage}%")
        
        advice.extend([
            "",
            "Popular Australian investment options:",
            "• ASX ETFs: VAS (Australian shares), VGS (International shares), NDQ (NASDAQ)",
            "• Individual ASX stocks: Major banks (CBA, ANZ), miners (BHP, RIO)",
            "• Term deposits: 4.0-4.5% guaranteed returns from major banks",
            "• Property: Direct investment or REITs via ETFs like VAP",
            "",
            "Key considerations:",
            "• Diversification across asset classes and geography",
            "• Consider franking credits on Australian dividend-paying stocks",
            "• Review investment fees - aim for low-cost ETFs (<0.30% p.a.)",
            "• Dollar-cost averaging for regular investment contributions"
        ])
        
        # Add context-specific information if available
        if context:
            advice.append("\nBased on current market information:")
            for ctx in context[:2]:
                if len(ctx) > 100:
                    advice.append(f"• {ctx[:150]}...")
        
        return advice
    
    def _precious_metals_advice(self, context: List[str]) -> List[str]:
        """Generate precious metals specific advice."""
        advice = ["🥇 Precious Metals Investment in Australia:"]
        
        # Add current price information from context if available
        current_info = []
        for ctx in context:
            if any(metal in ctx.lower() for metal in ['gold', 'silver', 'platinum']):
                current_info.append(ctx[:200] + "...")
        
        if current_info:
            advice.append("\nCurrent market conditions:")
            advice.extend([f"• {info}" for info in current_info[:2]])
        
        advice.extend([
            "",
            "Investment options for Australian precious metals exposure:",
            "",
            "Physical Metals:",
            "• Gold/silver bullion from Perth Mint or ABC Bullion",
            "• Storage costs: $200-500 annually for allocated storage",
            "• No GST on gold, 10% GST on silver",
            "",
            "ASX Mining Stocks:",
            "• Gold producers: Northern Star (NST), Evolution Mining (EVN)",
            "• Diversified miners: BHP, Rio Tinto with metals exposure",
            "• Higher volatility but potential for dividends",
            "",
            "ETFs and Funds:",
            "• International gold ETFs (limited Australian options)",
            "• Precious metals managed funds",
            "",
            "Portfolio allocation: Consider 5-10% in precious metals for diversification."
        ])
        
        return advice
    
    def _general_financial_advice(self, query: str, context: List[str], calculations: Dict) -> List[str]:
        """Generate general financial advice."""
        advice = [f"💡 Financial Guidance:"]
        
        # Use context to provide relevant information
        if context:
            advice.append("\nBased on current Australian financial information:")
            for ctx in context[:3]:
                if len(ctx) > 50:
                    advice.append(f"• {ctx[:200]}...")
        
        advice.extend([
            "",
            "General Australian financial planning principles:",
            "• Build emergency fund first (3-6 months expenses)",
            "• Maximize employer super contributions",
            "• Pay down high-interest debt before investing",
            "• Diversify investments across asset classes",
            "• Consider tax-effective investment structures",
            "• Review and rebalance portfolio annually",
            "",
            "For specific advice tailored to your situation, consider consulting",
            "a licensed financial adviser."
        ])
        
        return advice
    
    def _extract_sources(self, search_results: Dict) -> List[str]:
        """Extract source information from search results."""
        sources = []
        metadatas = search_results.get('metadatas', [])
        
        for metadata_list in metadatas:
            for metadata in metadata_list:
                source = metadata.get('source', 'Unknown')
                category = metadata.get('category', '')
                if source not in sources:
                    sources.append(f"{source} ({category})" if category else source)
        
        return sources[:5]  # Limit to top 5 sources
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information."""
        try:
            collection_info = self.chroma_manager.get_collection_info()
            return {
                'status': 'healthy',
                'documents_count': collection_info.get('count', 0),
                'embedding_model': self.config.get('embedding.model_name'),
                'database_path': self.config.get('database.path'),
                'last_updated': collection_info.get('last_updated', 'Unknown')
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
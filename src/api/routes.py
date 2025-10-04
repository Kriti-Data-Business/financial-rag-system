"""API routes for the RAG system."""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import hashlib

from src.data.database.vector_store import VectorStore
from src.data.processors.tabular_processor import TabularProcessor
from src.models.llm import LLMModel
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Initialize components
try:
    vector_store = VectorStore()
    llm_model = LLMModel()
    tabular_processor = TabularProcessor()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    vector_store = None
    llm_model = None
    tabular_processor = None


# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: Optional[List[dict]] = None


class FinancialProfileRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    monthly_income: float = Field(gt=0)
    monthly_expenses: Optional[float] = None
    current_savings: Optional[float] = 0
    dependents: int = Field(default=0, ge=0)
    has_health_insurance: bool = False
    has_life_insurance: bool = False
    debt: Optional[float] = 0
    risk_tolerance: Optional[str] = "moderate"


class FinancialGuidanceResponse(BaseModel):
    guidance: str
    calculations: dict
    priority_actions: List[str]
    disclaimer: str


# Routes
@router.get("/system/info")
async def get_system_info():
    """Get system configuration information."""
    return {
        "ollama_model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "status": "healthy" if llm_model else "unhealthy",
        "documents_indexed": vector_store.collection.count() if vector_store else 0
    }


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document."""
    try:
        file_ext = Path(file.filename).suffix.lower()
        temp_path = Path(settings.data_raw_path) / file.filename
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        temp_path.write_bytes(content)
        
        if file_ext in ['.csv', '.xlsx', '.xls']:
            chunks = tabular_processor.process_file(str(temp_path))
        else:
            text = content.decode('utf-8')
            chunk_size = settings.chunk_size
            chunks = []
            
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                chunk_id = hashlib.md5(f"{file.filename}_{i}".encode()).hexdigest()
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_text,
                    'source_document': file.filename
                })
        
        vector_store.add_documents(chunks)
        
        return {
            "filename": file.filename,
            "status": "success",
            "chunks_created": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))


@router.post("/documents/upload-directory")
async def upload_directory(directory_path: str):
    """Process all files in a directory."""
    try:
        path = Path(directory_path)
        if not path.exists():
            raise HTTPException(404, f"Directory not found: {directory_path}")
        
        all_chunks = tabular_processor.process_directory(directory_path)
        if not all_chunks:
            raise HTTPException(404, "No supported files found")
        
        vector_store.add_documents(all_chunks)
        
        return {
            "status": "success",
            "directory": directory_path,
            "chunks_created": len(all_chunks)
        }
    except Exception as e:
        logger.error(f"Directory upload error: {e}")
        raise HTTPException(500, str(e))


@router.post("/financial-guidance", response_model=FinancialGuidanceResponse)
async def get_financial_guidance(profile: FinancialProfileRequest):
    """Provide general financial planning guidance."""
    if not llm_model:
        raise HTTPException(503, "LLM not initialized")
    
    try:
        # Calculate metrics
        annual_income = profile.monthly_income * 12
        estimated_expenses = profile.monthly_expenses or (profile.monthly_income * 0.65)
        recommended_emergency_fund = estimated_expenses * 6
        
        life_insurance_multiplier = 10 if profile.dependents > 0 else 5
        recommended_life_insurance = annual_income * life_insurance_multiplier
        
        monthly_savings_target = profile.monthly_income * 0.20
        years_to_retirement = max(65 - profile.age, 0)
        
        # Build context
        context = f"""
FINANCIAL PROFILE:
- Age: {profile.age} years
- Monthly Income: ${profile.monthly_income:,.2f}
- Annual Income: ${annual_income:,.2f}
- Current Savings: ${profile.current_savings:,.2f}
- Debt: ${profile.debt:,.2f}
- Dependents: {profile.dependents}
- Years to Retirement: {years_to_retirement}

RECOMMENDATIONS:
- Emergency Fund Target: ${recommended_emergency_fund:,.2f}
- Life Insurance Target: ${recommended_life_insurance:,.2f}
- Monthly Savings Target: ${monthly_savings_target:,.2f}
"""

        query = f"""Provide financial guidance for a {profile.age}-year-old earning ${profile.monthly_income:,.0f}/month.
        
Include:
1. Priority actions
2. Emergency fund strategy
3. Insurance recommendations
4. Savings approach
5. Specific action steps

Be specific with numbers and actionable."""

        system_prompt = """You are a financial education assistant. Provide general guidance based on established principles.

REQUIREMENTS:
- Start with "Based on your profile..."
- Use specific dollar amounts
- Prioritize by urgency
- End with disclaimer about consulting professionals

DO NOT recommend specific products or predict returns."""

        guidance = llm_model.generate_response(query, context, system_prompt)
        
        # Determine priorities
        priority_actions = []
        
        if not profile.has_health_insurance:
            priority_actions.append("Get health insurance immediately")
        
        if profile.current_savings < estimated_expenses * 3:
            priority_actions.append(f"Build emergency fund to ${estimated_expenses * 3:,.0f}")
        
        if profile.debt > 0:
            priority_actions.append(f"Create debt payoff plan for ${profile.debt:,.0f}")
        
        if profile.dependents > 0 and not profile.has_life_insurance:
            priority_actions.append(f"Get term life insurance (~${recommended_life_insurance:,.0f})")
        
        if not priority_actions:
            priority_actions.append("Continue current plan and review annually")
        
        calculations = {
            "annual_income": round(annual_income, 2),
            "emergency_fund_target": round(recommended_emergency_fund, 2),
            "life_insurance_target": round(recommended_life_insurance, 2),
            "monthly_savings_target": round(monthly_savings_target, 2),
            "years_to_retirement": years_to_retirement
        }
        
        return FinancialGuidanceResponse(
            guidance=guidance,
            calculations=calculations,
            priority_actions=priority_actions[:5],
            disclaimer="This is educational guidance only, not financial advice. Consult licensed professionals for personalized recommendations."
        )
    
    except Exception as e:
        logger.error(f"Financial guidance error: {e}")
        raise HTTPException(500, str(e))


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    if not llm_model:
        raise HTTPException(503, "LLM not initialized. Check server logs.")
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        search_results = vector_store.search(request.query, request.top_k)
        
        if not search_results:
            return QueryResponse(
                query=request.query,
                answer="No documents found. Please upload documents first.",
                sources=[]
            )
        
        # Prepare context (limit size)
        context_parts = []
        total_length = 0
        max_context = 2000
        
        for r in search_results:
            if total_length + len(r['content']) > max_context:
                break
            context_parts.append(r['content'])
            total_length += len(r['content'])
        
        context = "\n\n".join(context_parts)
        
        logger.info(f"Context prepared: {len(context)} characters")
        
        # Generate answer
        answer = llm_model.generate_response(request.query, context)
        
        logger.info("Query completed successfully")
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=[{
                'content': r['content'][:300],
                'score': r['score']
            } for r in search_results[:3]]
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(500, f"Error processing query: {str(e)}")


@router.get("/collection/stats")
async def get_stats():
    """Get collection statistics."""
    return vector_store.get_collection_stats()
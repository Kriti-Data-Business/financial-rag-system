# app.py
# FastAPI application for Australian Financial RAG System

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import uvicorn
import logging
from pathlib import Path

from src.models.rag_system import AustralianFinancialRAGSystem
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger('api', 'logs/api.log')

# Initialize FastAPI app
app = FastAPI(
    title="Australian Financial RAG API",
    description="AI-powered Australian financial advice system using RAG",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None

# Request models
class UserProfile(BaseModel):
    age: Optional[int] = Field(None, ge=18, le=100, description="User's age")
    annual_salary: Optional[float] = Field(None, ge=0, description="Annual salary in AUD")
    monthly_expenses: Optional[float] = Field(None, ge=0, description="Monthly expenses in AUD")
    dependents: Optional[int] = Field(0, ge=0, description="Number of dependents")
    risk_tolerance: Optional[str] = Field("moderate", description="Risk tolerance: conservative, moderate, aggressive")
    current_super_balance: Optional[float] = Field(None, ge=0, description="Current superannuation balance")
    retirement_age: Optional[int] = Field(67, ge=55, le=75, description="Planned retirement age")

class AdviceRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Financial question")
    user_profile: Optional[UserProfile] = None
    num_results: Optional[int] = Field(5, ge=1, le=10, description="Number of documents to retrieve")

class InitializeRequest(BaseModel):
    data_path: Optional[str] = Field("data/processed/documents", description="Path to processed documents")

# Response models
class AdviceResponse(BaseModel):
    response: str
    context: List[str]
    financial_calculations: Dict[str, Any]
    sources: List[str]
    metadata: Dict[str, Any]

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag_system
    try:
        logger.info("Initializing Australian Financial RAG System...")
        rag_system = AustralianFinancialRAGSystem(config_path="config/production.yaml")
        logger.info("RAG System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Australian Financial RAG API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "advice": "/api/v1/advice",
            "health": "/health",
            "stats": "/api/v1/stats",
            "initialize": "/api/v1/initialize"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if rag_system is None:
            return {"status": "initializing", "ready": False}
        
        stats = rag_system.get_system_stats()
        return {
            "status": stats.get('status', 'unknown'),
            "ready": True,
            "documents_count": stats.get('documents_count', 0),
            "embedding_model": stats.get('embedding_model', 'unknown')
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "ready": False}

@app.post("/api/v1/advice", response_model=AdviceResponse)
async def get_financial_advice(request: AdviceRequest):
    """
    Get financial advice based on user query and profile.
    
    Args:
        request: Advice request with query and optional user profile
        
    Returns:
        Comprehensive financial advice with calculations and sources
    """
    try:
        if rag_system is None:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        # Convert user profile to dict
        user_profile_dict = None
        if request.user_profile:
            user_profile_dict = request.user_profile.dict(exclude_none=True)
        
        # Get advice
        result = rag_system.get_financial_advice(
            query=request.query,
            user_profile=user_profile_dict,
            num_results=request.num_results
        )
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return AdviceResponse(
            response=result.get('response', ''),
            context=result.get('context', []),
            financial_calculations=result.get('financial_calculations', {}),
            sources=result.get('sources', []),
            metadata={
                'enhanced_query': result.get('enhanced_query', ''),
                'num_sources': len(result.get('sources', []))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing advice request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_system_stats():
    """Get system statistics and health information."""
    try:
        if rag_system is None:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        return rag_system.get_system_stats()
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/initialize")
async def initialize_knowledge_base(
    request: InitializeRequest,
    background_tasks: BackgroundTasks
):
    """
    Initialize or reinitialize the knowledge base with documents.
    
    Args:
        request: Initialization request with data path
        background_tasks: FastAPI background tasks
        
    Returns:
        Status of initialization
    """
    try:
        if rag_system is None:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        # Add initialization to background tasks
        background_tasks.add_task(
            rag_system.initialize_knowledge_base,
            request.data_path
        )
        
        return {
            "status": "initialization_started",
            "message": f"Knowledge base initialization started with path: {request.data_path}"
        }
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/calculate/emergency-fund")
async def calculate_emergency_fund(monthly_expenses: float, months: int = 6):
    """Calculate emergency fund requirement."""
    try:
        if rag_system is None:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        return rag_system.financial_calculator.calculate_emergency_fund(
            monthly_expenses, months
        )
    except Exception as e:
        logger.error(f"Error calculating emergency fund: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/calculate/super-guarantee")
async def calculate_super_guarantee(annual_salary: float):
    """Calculate superannuation guarantee."""
    try:
        if rag_system is None:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        return rag_system.financial_calculator.calculate_super_guarantee(annual_salary)
    except Exception as e:
        logger.error(f"Error calculating super guarantee: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/calculate/investment-allocation")
async def calculate_investment_allocation(
    disposable_income: float,
    age: int,
    risk_tolerance: str = "moderate"
):
    """Calculate recommended investment allocation."""
    try:
        if rag_system is None:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        return rag_system.financial_calculator.calculate_investment_allocation(
            disposable_income, age, risk_tolerance
        )
    except Exception as e:
        logger.error(f"Error calculating investment allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

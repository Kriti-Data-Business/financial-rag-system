"""
Australian Financial RAG System - Streamlit Application
Complete web interface for personalized financial planning
"""

import streamlit as st
import sys
import os
from src.data_sources import DATA_SOURCES

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from embeddings import EmbeddingModel
from vector_store import VectorStore
from llm_handler import LocalLLMHandler
from financial_calculator import AustralianFinancialCalculator
from rag_system import AustralianFinancialRAG
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Australian Financial RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Australian Financial RAG System v1.0"
    }
)

# =============================================================================
# CUSTOM CSS STYLING - PROFESSIONAL ENTERPRISE THEME
# =============================================================================

st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 4px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border-radius: 4px;
        transition: all 0.2s ease;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    
    .stButton>button:hover {
        background-color: #34495e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        border-left: 3px solid #6c757d;
        margin: 1rem 0;
        color: #495057;
    }
    
    .success-box {
        background-color: #f1f8f4;
        padding: 1rem;
        border-radius: 4px;
        border-left: 3px solid #28a745;
        margin: 1rem 0;
        color: #155724;
    }
    
    .warning-box {
        background-color: #fff8f0;
        padding: 1rem;
        border-radius: 4px;
        border-left: 3px solid #ffc107;
        margin: 1rem 0;
        color: #856404;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    .assistant-message {
        background-color: white;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border-left: 3px solid #2c3e50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: white;
    }
    
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e9ecef;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: white;
        padding: 0.5rem;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding-left: 24px;
        padding-right: 24px;
        background-color: #f8f9fa;
        border-radius: 4px;
        color: #495057;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50;
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 4px;
        border: 1px solid #e9ecef;
        font-weight: 500;
        color: #2c3e50;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #1a1a1a;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6c757d;
        font-weight: 500;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #2c3e50;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 4px;
        border: 1px solid #ced4da;
        font-size: 0.95rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    
    /* Table styling */
    .dataframe {
        border: 1px solid #e9ecef !important;
        border-radius: 4px;
    }
    
    .dataframe th {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.8rem !important;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.chat_history = []
    st.session_state.user_profile = {}
    st.session_state.models_loaded = False
    st.session_state.current_query = ""

# =============================================================================
# MODEL LOADING (CACHED)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all models with caching"""
    try:
        embedding_model = EmbeddingModel()
        vector_store = VectorStore()
        vector_store.create_collection(embedding_model)
        llm_handler = LocalLLMHandler()
        calculator = AustralianFinancialCalculator()
        rag_system = AustralianFinancialRAG(vector_store, embedding_model, llm_handler, calculator)
        return rag_system, calculator, True, None
    except Exception as e:
        return None, None, False, str(e)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_currency(amount):
    """Format currency with AUD symbol"""
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format percentage"""
    return f"{value*100:.1f}%"

def create_gauge_chart(value, max_value, title):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16, 'color': '#1a1a1a', 'family': 'Inter'}},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, max_value*0.33], 'color': "#f8f9fa"},
                {'range': [max_value*0.33, max_value*0.67], 'color': "#e9ecef"},
                {'range': [max_value*0.67, max_value], 'color': "#dee2e6"}
            ],
            'threshold': {
                'line': {'color': "#dc3545", 'width': 3},
                'thickness': 0.75,
                'value': max_value*0.9
            }
        }
    ))
    fig.update_layout(
        height=250,
        font=dict(family='Inter', color='#495057'),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig

def get_risk_profile_color(risk_profile):
    """Get color based on risk profile"""
    colors = {
        'Conservative': '#6c757d',
        'Balanced': '#495057',
        'Moderate Growth': '#343a40',
        'Aggressive Growth': '#212529'
    }
    return colors.get(risk_profile, '#6c757d')

def safe_dataframe_display(df, use_container_width=True):
    numeric_cols = df.select_dtypes(include=['number']).columns
    format_dict = {col: '${:,.2f}' for col in numeric_cols}
    if format_dict:
        st.dataframe(df.style.format(format_dict), use_container_width=use_container_width)
    else:
        st.dataframe(df, use_container_width=use_container_width)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    
    # Header section
    st.markdown('<div class="main-header">Australian Financial RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Personalized Financial Planning</div>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models... This may take a few minutes on first run."):
        rag_system, calculator, success, error = load_models()
    
    if not success:
        st.error(f"Error loading models: {error}")
        st.info("""
        **Please ensure you have:**
        1. Run `python scripts/initialize_data.py`
        2. Installed all requirements: `pip install -r requirements.txt`
        3. Have sufficient disk space (~4GB for models)
        """)
        return
    
    st.session_state.models_loaded = True
    
    # =============================================================================
    # SIDEBAR - USER PROFILE
    # =============================================================================
    
    with st.sidebar:
        st.markdown("### USER PROFILE")
        st.markdown("---")
        
        # Personal Information
        with st.expander("Personal Details", expanded=True):
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
                help="Your current age"
            )
            
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married/De facto", "Separated", "Divorced", "Widowed"]
            )
            
            dependents = st.number_input(
                "Number of Dependents",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )
        
        # Income Details
        with st.expander("Income Details", expanded=True):
            income_type = st.radio(
                "Income Period",
                ["Monthly", "Annual"],
                horizontal=True
            )
            
            if income_type == "Monthly":
                monthly_income = st.number_input(
                    "Monthly Income (AUD)",
                    min_value=0,
                    value=5000,
                    step=500,
                    help="Your gross monthly income before tax"
                )
                annual_income = monthly_income * 12
            else:
                annual_income = st.number_input(
                    "Annual Income (AUD)",
                    min_value=0,
                    value=60000,
                    step=5000,
                    help="Your gross annual income before tax"
                )
                monthly_income = annual_income / 12
            
            income_source = st.multiselect(
                "Income Sources",
                ["Employment", "Self-Employment", "Investments", "Rental", "Other"],
                default=["Employment"]
            )
        
        # Current Financial Position
        with st.expander("Current Position", expanded=True):
            savings = st.number_input(
                "Current Savings (AUD)",
                min_value=0,
                value=0,
                step=1000,
                help="Total liquid savings and cash"
            )
            
            super_balance = st.number_input(
                "Superannuation Balance (AUD)",
                min_value=0,
                value=0,
                step=5000,
                help="Current super balance"
            )
            
            debt = st.number_input(
                "Total Debt (AUD)",
                min_value=0,
                value=0,
                step=1000,
                help="Total outstanding debt (excluding mortgage)"
            )
        
        # Financial Goals
        with st.expander("Financial Goals", expanded=True):
            goal_category = st.selectbox(
                "Primary Goal",
                [
                    "Wealth Accumulation",
                    "Retirement Planning",
                    "First Home Purchase",
                    "Debt Reduction",
                    "Emergency Fund",
                    "Investment Growth",
                    "Tax Optimization",
                    "Other"
                ]
            )
            
            goal = st.text_area(
                "Describe Your Goal",
                placeholder="e.g., Save $1M in 10 years, Plan for retirement at 60, Buy first home in 5 years",
                value="General financial advice",
                height=100
            )
            
            time_horizon = st.slider(
                "Time Horizon (Years)",
                min_value=1,
                max_value=40,
                value=10,
                help="When do you want to achieve this goal?"
            )
        
        # Save Profile Button
        st.markdown("---")
        if st.button("SAVE PROFILE", use_container_width=True):
            st.session_state.user_profile = {
                'age': age,
                'marital_status': marital_status,
                'dependents': dependents,
                'monthly_income': monthly_income,
                'annual_income': annual_income,
                'income_source': income_source,
                'savings': savings,
                'super_balance': super_balance,
                'debt': debt,
                'goal': goal,
                'goal_category': goal_category,
                'time_horizon': time_horizon
            }
            st.success("Profile saved successfully")
        
        # Quick Insights in Sidebar
        if annual_income > 0:
            st.markdown("---")
            st.markdown("### QUICK INSIGHTS")
            
            after_tax = calculator.calculate_after_tax_income(annual_income)
            tax_paid = annual_income - after_tax
            
            # Compact metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "After Tax",
                    format_currency(after_tax/12),
                    delta=f"{format_percentage((after_tax/annual_income))} take home"
                )
                st.metric(
                    "Tax Rate",
                    format_percentage(tax_paid/annual_income),
                    delta=f"-{format_currency(tax_paid/12)}/mo",
                    delta_color="inverse"
                )
            with col2:
                st.metric(
                    "Super (11%)",
                    format_currency(annual_income*0.11/12),
                    delta=f"{format_currency(annual_income*0.11)}/yr"
                )
                allocation = calculator.calculate_investment_allocation(age)
                st.metric(
                    "Risk Profile",
                    allocation['risk_profile'].split()[0],
                    delta=f"Age {age}"
                )
    
    # =============================================================================
    # MAIN CONTENT TABS
    # =============================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "AI Advisor Chat",
        "Financial Calculators",
        "Portfolio Analysis",
        "Learning Center",
        "About System"
    ])
    
    # =============================================================================
    # TAB 1: AI ADVISOR CHAT
    # =============================================================================
    
    with tab1:
        st.header("Chat with Your AI Financial Advisor")
        st.markdown("Ask questions about investments, superannuation, tax, emergency funds, and more")
        
        # Example queries section
        st.subheader("Example Questions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Savings & Emergency Fund**")
            if st.button("How much should I save for emergency fund?", key="ex1"):
                st.session_state.current_query = "How much should I save for emergency fund based on my income and expenses?"
            
            if st.button("How much can I save per month?", key="ex2"):
                st.session_state.current_query = f"I earn ${monthly_income:,.0f} per month. How much should I save?"
            
            st.markdown("**Investment Planning**")
            if st.button("What's my recommended investment allocation?", key="ex3"):
                st.session_state.current_query = f"What's the recommended investment allocation for a {age} year old?"
            
            if st.button("Should I invest in international stocks?", key="ex4"):
                st.session_state.current_query = "Should I invest in international stocks like VGS or focus on Australian stocks?"
        
        with col2:
            st.markdown("**Superannuation**")
            if st.button("Should I salary sacrifice to super?", key="ex5"):
                st.session_state.current_query = f"Should I salary sacrifice to super with my income of ${annual_income:,.0f}?"
            
            if st.button("How much will my super be worth?", key="ex6"):
                st.session_state.current_query = f"I'm {age} with ${super_balance:,.0f} in super. How much will I have at retirement?"
            
            st.markdown("**Goal Planning**")
            if st.button("Help me reach my financial goal", key="ex7"):
                st.session_state.current_query = f"Help me create a detailed plan to: {goal}"
            
            if st.button("First home buyer strategy", key="ex8"):
                st.session_state.current_query = "What's the best strategy for first home buyers in Australia?"
        
        st.markdown("---")
        
        # Chat input section
        col1, col2 = st.columns([4, 1])
        with col1:
            user_query = st.text_input(
                "Type your question:",
                value=st.session_state.current_query,
                placeholder="e.g., How should I invest $50,000?",
                key="chat_input_main"
            )
        with col2:
            send_button = st.button("GET ADVICE", use_container_width=True)
        
        # Process query
        if send_button or (user_query and user_query != st.session_state.current_query):
            if not st.session_state.user_profile:
                st.warning("Please fill in your profile in the sidebar first")
            elif not user_query:
                st.warning("Please enter a question")
            else:
                with st.spinner("Analyzing your situation and finding the best advice..."):
                    try:
                        # Enhance query
                        enhanced_query = rag_system.enhance_query(
                            st.session_state.user_profile,
                            user_query
                        )
                        
                        # Progress indicators
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Searching knowledge base...")
                        progress_bar.progress(20)
                        
                        # Retrieve documents
                        retrieved_docs = rag_system.retrieve_relevant_docs(enhanced_query, top_k=5)
                        
                        status_text.text("Calculating financial metrics...")
                        progress_bar.progress(40)
                        
                        # Calculate metrics
                        financial_metrics = rag_system.calculate_financial_metrics(
                            st.session_state.user_profile
                        )
                        
                        status_text.text("Generating personalized advice...")
                        progress_bar.progress(60)
                        
                        # Generate response
                        response = rag_system.generate_response(
                            st.session_state.user_profile,
                            user_query,
                            retrieved_docs,
                            financial_metrics
                        )
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Display response
                        st.markdown("---")
                        st.subheader("Your Personalized Financial Advice")
                        
                        st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
                        
                        # Show key metrics used
                        with st.expander("Key Financial Metrics Used in This Advice"):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            if 'after_tax_monthly' in financial_metrics:
                                col1.metric(
                                    "Monthly After-Tax",
                                    format_currency(financial_metrics['after_tax_monthly'])
                                )
                                col2.metric(
                                    "Tax Rate",
                                    format_percentage(financial_metrics['effective_tax_rate'])
                                )
                                col3.metric(
                                    "Super Contribution",
                                    format_currency(financial_metrics['super_guarantee'])
                                )
                            
                            col4.metric(
                                "Risk Profile",
                                financial_metrics['allocation']['risk_profile']
                            )
                        
                        # Show retrieved sources
                        with st.expander("Sources & References Used"):
                            for i, (doc, metadata) in enumerate(zip(
                                retrieved_docs['documents'][0],
                                retrieved_docs['metadatas'][0]
                            )):
                                st.markdown(f"**Source {i+1}: {metadata['title']}**")
                                st.caption(f"Category: {metadata['category']} | Source: {metadata['source']}")
                                with st.container():
                                    st.text(doc[:400] + "..." if len(doc) > 400 else doc)
                                if i < len(retrieved_docs['documents'][0]) - 1:
                                    st.markdown("---")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'query': user_query,
                            'response': response,
                            'metrics': financial_metrics
                        })
                        
                        # Clear current query
                        st.session_state.current_query = ""
                        
                    except Exception as e:
                        st.error(f"Error generating advice: {str(e)}")
                        st.info("Please try rephrasing your question or check if models are loaded correctly.")
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("Recent Conversations")
            
            # Show last 5 conversations
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"{chat['timestamp']}: {chat['query'][:80]}..."):
                    st.markdown(f"**Your Question:**")
                    st.info(chat['query'])
                    st.markdown(f"**Advisor Response:**")
                    st.success(chat['response'])
            
            if st.button("CLEAR CHAT HISTORY"):
                st.session_state.chat_history = []
                st.rerun()
    
    # =============================================================================
    # TAB 2: FINANCIAL CALCULATORS
    # =============================================================================
    
    with tab2:
        st.header("Financial Calculators")
        st.markdown("Interactive tools to plan your financial future")
        
        calc_tab1, calc_tab2, calc_tab3, calc_tab4 = st.tabs([
            "Emergency Fund",
            "Salary Sacrifice",
            "Investment Growth",
            "Home Savings"
        ])
        
        # Emergency Fund Calculator
        with calc_tab1:
            st.subheader("Emergency Fund Calculator")
            st.markdown("Calculate how much you should save for unexpected expenses")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                monthly_expenses = st.number_input(
                    "Monthly Expenses (AUD)",
                    min_value=0,
                    value=3000,
                    step=100,
                    help="Include rent, food, utilities, transport, insurance"
                )
                
                months = st.slider(
                    "Months of Coverage",
                    min_value=3,
                    max_value=12,
                    value=6,
                    help="Recommended: 3-6 months for dual income, 6-12 for single income"
                )
                
                current_savings = st.number_input(
                    "Current Emergency Savings (AUD)",
                    min_value=0,
                    value=0,
                    step=500
                )
            
            with col2:
                emergency_fund = calculator.calculate_emergency_fund(monthly_expenses, months)
                remaining = max(0, emergency_fund - current_savings)
                progress = min(100, (current_savings / emergency_fund * 100) if emergency_fund > 0 else 0)
                
                st.metric(
                    "Target Emergency Fund",
                    format_currency(emergency_fund),
                    delta=f"{months} months coverage"
                )
                
                st.metric(
                    "Still Needed",
                    format_currency(remaining),
                    delta=f"{progress:.0f}% complete"
                )
                
                if progress >= 100:
                    st.success("Goal Achieved")
                elif progress >= 50:
                    st.info("Halfway there")
                else:
                    st.warning("Keep saving")
            
            # Visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Current Savings',
                x=['Emergency Fund'],
                y=[current_savings],
                marker_color='#2c3e50',
                text=[format_currency(current_savings)],
                textposition='auto'
            ))
            
            fig.add_trace(go.Bar(
                name='Remaining',
                x=['Emergency Fund'],
                y=[remaining],
                marker_color='#95a5a6',
                text=[format_currency(remaining)],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"{months}-Month Emergency Fund Progress",
                yaxis_title="Amount (AUD)",
                barmode='stack',
                height=400,
                font=dict(family='Inter'),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Savings plan
            if remaining > 0:
                st.subheader("Savings Plan")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Save in 6 months",
                        format_currency(remaining/6) + "/mo"
                    )
                with col2:
                    st.metric(
                        "Save in 12 months",
                        format_currency(remaining/12) + "/mo"
                    )
                with col3:
                    st.metric(
                        "Save in 24 months",
                        format_currency(remaining/24) + "/mo"
                    )
        
        # Salary Sacrifice Calculator
        with calc_tab2:
            st.subheader("Salary Sacrifice to Super Calculator")
            st.markdown("See how much tax you can save by contributing to superannuation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                salary = st.number_input(
                    "Annual Salary (AUD)",
                    min_value=0,
                    value=80000,
                    step=5000
                )
                
                sacrifice = st.slider(
                    "Salary Sacrifice Amount (AUD)",
                    min_value=0,
                    max_value=30000,
                    value=10000,
                    step=1000,
                    help="Maximum concessional cap is $30,000 per year"
                )
                
                if sacrifice > 30000:
                    st.warning("Exceeds concessional cap of $30,000")
            
            with col2:
                if salary > 0 and sacrifice > 0:
                    benefit = calculator.calculate_salary_sacrifice_benefit(salary, sacrifice)
                    
                    st.metric(
                        "Annual Tax Saved",
                        format_currency(benefit['tax_saved']),
                        delta="More in your pocket"
                    )
                    
                    st.metric(
                        "Super Tax Paid",
                        format_currency(benefit['super_tax']),
                        delta="15% rate",
                        delta_color="inverse"
                    )
                    
                    st.metric(
                        "Net Benefit",
                        format_currency(benefit['net_benefit']),
                        delta="Per year"
                    )
            
            if salary > 0 and sacrifice > 0:
                # Comparison visualization
                fig = go.Figure()
                
                without_sacrifice = calculator.calculate_income_tax(salary)
                with_sacrifice = calculator.calculate_income_tax(salary - sacrifice)
                
                fig.add_trace(go.Bar(
                    name='Without Sacrifice',
                    x=['Income Tax', 'Take Home', 'Super'],
                    y=[
                        without_sacrifice,
                        salary - without_sacrifice,
                        salary * 0.11
                    ],
                    marker_color='#95a5a6'
                ))
                
                fig.add_trace(go.Bar(
                    name='With Sacrifice',
                    x=['Income Tax', 'Take Home', 'Super'],
                    y=[
                        with_sacrifice,
                        salary - sacrifice - with_sacrifice,
                        salary * 0.11 + sacrifice - benefit['super_tax']
                    ],
                    marker_color='#2c3e50'
                ))
                
                fig.update_layout(
                    title="Tax Comparison: With vs Without Salary Sacrifice",
                    yaxis_title="Amount (AUD)",
                    barmode='group',
                    height=400,
                    font=dict(family='Inter'),
                    paper_bgcolor='white',
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown
                with st.expander("Detailed Breakdown"):
                    df = pd.DataFrame({
                        'Scenario': ['Without Sacrifice', 'With Sacrifice'],
                        'Gross Income': [salary, salary],
                        'Salary Sacrifice': [0, sacrifice],
                        'Taxable Income': [salary, salary - sacrifice],
                        'Income Tax': [without_sacrifice, with_sacrifice],
                        'Take Home Pay': [salary - without_sacrifice, salary - sacrifice - with_sacrifice],
                        'Super Balance': [salary * 0.11, salary * 0.11 + sacrifice - benefit['super_tax']],
                        'Total Value': [
                            salary - without_sacrifice + salary * 0.11,
                            salary - sacrifice - with_sacrifice + salary * 0.11 + sacrifice - benefit['super_tax']
                        ]
                    })
                    safe_dataframe_display(df)
        
        # Investment Growth Calculator
        with calc_tab3:
            st.subheader("Investment Growth Calculator")
            st.markdown("Project your wealth accumulation over time with compound interest")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                principal = st.number_input(
                    "Starting Amount (AUD)",
                    min_value=0,
                    value=10000,
                    step=1000
                )
                
                monthly_contrib = st.number_input(
                    "Monthly Contribution (AUD)",
                    min_value=0,
                    value=500,
                    step=50
                )
                
                annual_return = st.slider(
                    "Expected Annual Return (%)",
                    min_value=0.0,
                    max_value=15.0,
                    value=7.0,
                    step=0.5,
                    help="Conservative: 5-6%, Balanced: 7-8%, Growth: 9-10%"
                ) / 100
                
                years = st.slider(
                    "Investment Period (Years)",
                    min_value=1,
                    max_value=40,
                    value=10
                )
            
            with col2:
                growth = calculator.calculate_compound_growth(
                    principal,
                    monthly_contrib,
                    annual_return,
                    years
                )
                
                st.metric(
                    "Final Amount",
                    format_currency(growth['final_amount']),
                    delta=f"In {years} years"
                )
                
                st.metric(
                    "Total Contributed",
                    format_currency(growth['total_contributed'])
                )
                
                st.metric(
                    "Investment Growth",
                    format_currency(growth['total_growth']),
                    delta=format_percentage(growth['total_growth']/growth['total_contributed'])
                )
            
            # Growth chart over time
            months = years * 12
            monthly_rate = annual_return / 12
            balances = []
            contributions = []
            
            for month in range(months + 1):
                fv_principal = principal * ((1 + monthly_rate) ** month)
                if monthly_rate > 0:
                    fv_contributions = monthly_contrib * (((1 + monthly_rate) ** month - 1) / monthly_rate)
                else:
                    fv_contributions = monthly_contrib * month
                balances.append(fv_principal + fv_contributions)
                contributions.append(principal + (monthly_contrib * month))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(months + 1)),
                y=contributions,
                mode='lines',
                name='Total Contributed',
                line=dict(color='#95a5a6', dash='dash'),
                fill=None
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(months + 1)),
                y=balances,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#2c3e50', width=3),
                fill='tonexty',
                fillcolor='rgba(44, 62, 80, 0.1)'
            ))
            
            fig.update_layout(
                title=f"Investment Growth Projection ({years} Years)",
                xaxis_title="Months",
                yaxis_title="Portfolio Value (AUD)",
                height=400,
                hovermode='x unified',
                font=dict(family='Inter'),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Year-by-year breakdown
            with st.expander("Year-by-Year Breakdown"):
                yearly_data = []
                for year in range(1, years + 1):
                    month = year * 12
                    fv_principal = principal * ((1 + monthly_rate) ** month)
                    if monthly_rate > 0:
                        fv_contributions = monthly_contrib * (((1 + monthly_rate) ** month - 1) / monthly_rate)
                    else:
                        fv_contributions = monthly_contrib * month
                    total = fv_principal + fv_contributions
                    contributed = principal + (monthly_contrib * month)
                    growth = total - contributed
                    
                    yearly_data.append({
                        'Year': year,
                        'Balance': total,
                        'Contributed': contributed,
                        'Growth': growth
                    })
                
                df_yearly = pd.DataFrame(yearly_data)
                safe_dataframe_display(df_yearly)
        
        # Home Savings Calculator
        with calc_tab4:
            st.subheader("First Home Savings Calculator")
            st.markdown("Calculate how long it will take to save for your first home deposit")
            
            col1, col2 = st.columns(2)
            
            with col1:
                property_price = st.number_input(
                    "Property Price (AUD)",
                    min_value=0,
                    value=600000,
                    step=50000
                )
                
                deposit_percentage = st.slider(
                    "Deposit Percentage (%)",
                    min_value=5,
                    max_value=20,
                    value=20,
                    help="5% with First Home Guarantee, 20% to avoid LMI"
                )
                
                current_home_savings = st.number_input(
                    "Current Savings (AUD)",
                    min_value=0,
                    value=50000,
                    step=5000
                )
                
                monthly_home_savings = st.number_input(
                    "Monthly Savings (AUD)",
                    min_value=0,
                    value=2000,
                    step=100
                )
            
            with col2:
                required_deposit = property_price * (deposit_percentage / 100)
                stamp_duty = property_price * 0.04
                other_costs = 10000
                total_needed = required_deposit + stamp_duty + other_costs
                
                remaining = max(0, total_needed - current_home_savings)
                months_needed = remaining / monthly_home_savings if monthly_home_savings > 0 else 0
                years_needed = months_needed / 12
                
                st.metric(
                    "Required Deposit",
                    format_currency(required_deposit),
                    delta=f"{deposit_percentage}% of price"
                )
                
                st.metric(
                    "Total Costs",
                    format_currency(total_needed),
                    delta="Inc. stamp duty & costs"
                )
                
                st.metric(
                    "Time to Save",
                    f"{years_needed:.1f} years",
                    delta=f"{months_needed:.0f} months"
                )
            
            # Progress visualization
            progress_pct = min(100, (current_home_savings / total_needed * 100) if total_needed > 0 else 0)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = progress_pct,
                title = {'text': "Savings Progress", 'font': {'size': 16, 'color': '#1a1a1a', 'family': 'Inter'}},
                delta = {'reference': 100, 'suffix': "%"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#2c3e50"},
                    'steps': [
                        {'range': [0, 33], 'color': "#f8f9fa"},
                        {'range': [33, 67], 'color': "#e9ecef"},
                        {'range': [67, 100], 'color': "#dee2e6"}
                    ],
                    'threshold': {
                        'line': {'color': "#dc3545", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                font=dict(family='Inter'),
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost breakdown
            st.subheader("Cost Breakdown")
            breakdown_data = {
                'Item': ['Deposit', 'Stamp Duty', 'Other Costs'],
                'Amount': [required_deposit, stamp_duty, other_costs]
            }
            df_breakdown = pd.DataFrame(breakdown_data)
            
            fig = go.Figure(data=[go.Pie(
                labels=df_breakdown['Item'],
                values=df_breakdown['Amount'],
                hole=.3,
                marker=dict(colors=['#2c3e50', '#495057', '#6c757d'])
            )])
            fig.update_layout(
                title="Home Purchase Costs",
                height=350,
                font=dict(family='Inter'),
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # =============================================================================
    # TAB 3: PORTFOLIO ANALYSIS
    # =============================================================================
    
    with tab3:
        st.header("Portfolio Analysis & Asset Allocation")
        
        if not st.session_state.user_profile or st.session_state.user_profile.get('annual_income', 0) == 0:
            st.info("Please complete your profile in the sidebar to see personalized portfolio analysis")
        else:
            profile = st.session_state.user_profile
            metrics = rag_system.calculate_financial_metrics(profile)
            allocation = metrics['allocation']
            
            # Overview section
            st.subheader("Financial Health Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Annual Income",
                    format_currency(profile['annual_income']),
                    delta=f"{format_currency(profile['monthly_income'])}/mo"
                )
            
            with col2:
                st.metric(
                    "After Tax Income",
                    format_currency(metrics['after_tax_annual']),
                    delta=format_percentage(metrics['after_tax_annual']/profile['annual_income'])
                )
            
            with col3:
                st.metric(
                    "Annual Tax",
                    format_currency(metrics['tax_paid']),
                    delta=format_percentage(metrics['effective_tax_rate']),
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    "Risk Profile",
                    allocation['risk_profile'],
                    delta=f"Age {profile['age']}"
                )
            
            st.markdown("---")
            
            # Asset allocation section
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Recommended Asset Allocation")
                st.caption(f"Optimized for age {profile['age']} - {allocation['risk_profile']}")
                
                allocation_data = {k: v for k, v in allocation.items() if k != 'risk_profile'}
                
                fig = go.Figure(data=[go.Pie(
                    labels=[k.replace('_', ' ').title() for k in allocation_data.keys()],
                    values=list(allocation_data.values()),
                    hole=.4,
                    marker=dict(colors=['#2c3e50', '#495057', '#6c757d', '#868e96'])
                )])
                
                fig.update_layout(
                    title="Asset Allocation",
                    height=400,
                    annotations=[dict(text=allocation['risk_profile'], x=0.5, y=0.5, font_size=14, showarrow=False)],
                    font=dict(family='Inter'),
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Allocation Details")
                
                for asset, pct in allocation_data.items():
                    st.markdown(f"**{asset.replace('_', ' ').title()}**")
                    st.progress(pct)
                    st.caption(f"{pct*100:.0f}% - {format_currency((profile['savings'] + profile['super_balance']) * pct)}")
                    st.markdown("")
            
            # Investment recommendations
            st.markdown("---")
            st.subheader("Recommended Investment Products")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Australian Equities")
                st.markdown("""
                - **VAS** - Vanguard Australian Shares
                - **A200** - BetaShares Australia 200
                - **STW** - SPDR S&P/ASX 200
                
                **Allocation:** {:.0%}  
                **Risk:** Medium to High  
                **Fees:** 0.04% - 0.13% p.a.
                """.format(allocation.get('australian_equities', 0)))
            
            with col2:
                st.markdown("### International Equities")
                st.markdown("""
                - **VGS** - Vanguard International Shares
                - **IVV** - iShares S&P 500
                - **NDQ** - BetaShares NASDAQ 100
                
                **Allocation:** {:.0%}  
                **Risk:** Medium to High  
                **Fees:** 0.04% - 0.48% p.a.
                """.format(allocation.get('international_equities', 0)))
            
            with col3:
                st.markdown("### Fixed Income / Bonds")
                st.markdown("""
                - **VAF** - Vanguard Australian Fixed Interest
                - **VGB** - Vanguard Australian Gov Bonds
                - **BILL** - BetaShares Australian Bank Bills
                
                **Allocation:** {:.0%}  
                **Risk:** Low to Medium  
                **Fees:** 0.20% - 0.25% p.a.
                """.format(allocation.get('bonds', 0)))
            
            # Tax efficiency section
            st.markdown("---")
            st.subheader("Tax Optimization Strategies")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Current Tax Situation")
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Take Home', 'Tax Paid'],
                    values=[metrics['after_tax_annual'], metrics['tax_paid']],
                    marker=dict(colors=['#2c3e50', '#95a5a6'])
                )])
                fig.update_layout(
                    title="Income Breakdown",
                    height=300,
                    font=dict(family='Inter'),
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("Effective Tax Rate", format_percentage(metrics['effective_tax_rate']))
                st.metric("Annual Tax Paid", format_currency(metrics['tax_paid']))
            
            with col2:
                st.markdown("#### Tax-Saving Opportunities")
                
                # Calculate potential savings
                max_concessional = 30000
                current_super = profile['annual_income'] * 0.11
                available_concessional = max_concessional - current_super
                
                if available_concessional > 0:
                    potential_benefit = calculator.calculate_salary_sacrifice_benefit(
                        profile['annual_income'],
                        min(available_concessional, 10000)
                    )
                    
                    st.success(f"You could save up to {format_currency(potential_benefit['net_benefit'])} annually by maximizing super contributions")
                    
                    st.markdown("""
                    **Strategies to Consider:**
                    1. Salary sacrifice to super
                    2. Spouse super contributions
                    3. Tax-effective investments (franked dividends)
                    4. Negative gearing (if applicable)
                    5. Claim all deductions
                    """)
                else:
                    st.info("You're already at the concessional super cap")
            
            # Superannuation projection
            st.markdown("---")
            st.subheader("Superannuation Projection")
            
            current_super = profile['super_balance']
            annual_contribution = profile['annual_income'] * 0.11
            years_to_retirement = max(1, 67 - profile['age'])
            
            # Project super balance
            super_projection = calculator.calculate_compound_growth(
                current_super,
                annual_contribution / 12,
                0.07,
                years_to_retirement
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Balance",
                    format_currency(current_super)
                )
            with learn_tab2:
            st.subheader("Australian Tax System 2024-25")
            
            with st.expander("Income Tax Brackets", expanded=True):
                tax_data = pd.DataFrame({
                    'Income Range': [
                        '$0 - $18,200',
                        '$18,201 - $45,000',
                        '$45,001 - $120,000',
                        '$120,001 - $180,000',
                        '$180,001+'
                    ],
                    'Tax Rate': ['0%', '19%', '32.5%', '37%', '45%'],
                    'Tax on Range': [
                        '$0',
                        '$5,092',
                        '$29,467',
                        '$51,667',
                        '$51,667 + 45%'
                    ]
                })
                st.table(tax_data)
                
                st.caption("**Plus Medicare Levy:** 2% of taxable income")
            
            with st.expander("Tax Offsets & Deductions"):
                st.markdown("""
                **Common Tax Offsets:**
                - Low Income Tax Offset (LITO): Up to $700
                - Low and Middle Income Tax Offset (LMITO): Varies
                - Spouse super contribution offset: Up to $540
                - Private health insurance rebate
                
                **Common Deductions:**
                - Work-related expenses
                - Home office expenses
                - Self-education expenses
                - Investment property expenses
                - Charitable donations
                - Income protection insurance
                """)
            
            with st.expander("Capital Gains Tax (CGT)"):
                st.markdown("""
                **CGT Basics:**
                - Tax on profit from selling assets
                - Applies to shares, property (not main residence), etc.
                - Added to assessable income
                
                **50% CGT Discount:**
                - Hold asset for > 12 months
                - Only pay tax on 50% of capital gain
                - Applies to individuals and trusts
                - Not available for companies
                
                **Example:**
                - Buy shares for $10,000
                - Sell after 2 years for $15,000
                - Capital gain: $5,000
                - Discount applied: $2,500 assessable
                - Tax at 32.5% rate: $812.50
                """)
        
        with learn_tab3:
            st.subheader("Investing Basics for Australians")
            
            with st.expander("Asset Classes", expanded=True):
                st.markdown("""
                **Australian Equities**
                - ASX-listed companies
                - Franking credits on dividends
                - Home bias consideration
                
                **International Equities**
                - Diversification beyond Australia
                - Currency exposure (AUD/USD)
                - Access to global growth
                
                **Fixed Income (Bonds)**
                - Government and corporate bonds
                - Lower risk, lower return
                - Income generation
                
                **Property**
                - Direct property investment
                - REITs (A-REITs) on ASX
                

# Financial RAG System
# README.md
# Australian Financial RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for Australian financial investment planning, featuring precious metals integration, ASX market data, and superannuation advice.

## 🎯 Project Overview

This RAG system provides personalized Australian financial advice by combining:
- **Real-time market data** (ASX stocks, precious metals prices in AUD)
- **Australian Bureau of Statistics** household income and superannuation data
- **Regulatory knowledge** (tax brackets, super contribution limits)
- **Investment options** (ETFs, term deposits, mining stocks, precious metals)

Built using the proven **Walert framework methodology** adapted for the financial domain.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Data Sources  │    │   RAG Pipeline   │    │  User Interface     │
│                 │    │                  │    │                     │
│ • ABS Statistics│───▶│ • Document       │───▶│ • Streamlit App     │
│ • ASX Prices    │    │   Processing     │    │ • Interactive Chat  │
│ • Precious      │    │ • Vector         │    │ • Financial         │
│   Metals        │    │   Embeddings     │    │   Calculators       │
│ • Financial     │    │ • ChromaDB       │    │ • Portfolio         │
│   News          │    │ • LLM Generation │    │   Visualization     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/australian-financial-rag-system.git
   cd australian-financial-rag-system
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the system**
   ```bash
   python scripts/setup_environment.py
   python scripts/collect_data.py --source all
   python scripts/initialize_rag.py
   ```

5. **Launch the application**
   ```bash
   streamlit run app/main.py
   ```

Visit `http://localhost:8501` to access the financial planning interface.

## 💰 Features

### Core Capabilities
- **Emergency Fund Calculation** - 6 months expenses rule with Australian context
- **Superannuation Optimization** - 11% guarantee, salary sacrifice benefits
- **Investment Allocation** - Age-based strategies with Australian assets
- **Tax-Effective Planning** - CGT discount, franked dividends, marginal rates
- **Precious Metals Analysis** - Gold, silver, platinum in AUD with mining stocks
- **ASX Market Integration** - Real-time ETF and stock prices

### Data Sources
- **Australian Bureau of Statistics (ABS)** - Household income, wealth distribution
- **Australian Prudential Regulation Authority (APRA)** - Superannuation statistics
- **Reserve Bank of Australia (RBA)** - Economic indicators, cash rates
- **ASX Market Data** - ETF prices, mining stocks, blue chips
- **Precious Metals APIs** - Live gold/silver prices in AUD
- **Financial News Feeds** - Australian Financial Review, ABC Finance

### Investment Universe
- **ETFs**: VAS, VGS, NDQ, A200, VAF, VAP
- **Stocks**: CBA, ANZ, BHP, RIO, NST, EVN
- **Precious Metals**: Physical gold/silver, mining companies
- **Cash Products**: Term deposits, high-yield savings
- **Property**: REITs, direct investment analysis

## 🛠️ Development

### VS Code Setup
The project includes comprehensive VS Code configuration:

1. **Open in VS Code**
   ```bash
   code .
   ```

2. **Install recommended extensions** (prompted automatically)

3. **Select Python interpreter**: `./venv/bin/python`

4. **Available debug configurations**:
   - 🚀 Run Streamlit App
   - 🧠 Test RAG System  
   - 📊 Run Data Collection
   - 💎 Collect Precious Metals Data
   - 📈 Collect ASX Data

### Project Structure
```
financial-rag-system/
├── src/                    # Source code
│   ├── models/            # RAG system, embeddings, financial calculator
│   ├── data/              # Data collectors, processors, database
│   ├── evaluation/        # Testing framework (Walert methodology)
│   ├── api/               # FastAPI interface (optional)
│   └── utils/             # Configuration, logging, helpers
├── app/                   # Streamlit web application
├── data/                  # Data storage (raw, processed, evaluation)
├── tests/                 # Pytest test suite
├── scripts/               # Automation scripts
├── notebooks/             # Jupyter analysis notebooks
└── docs/                  # Documentation
```

### Key Components

#### RAG System (`src/models/rag_system.py`)
- Main orchestrator for financial advice generation
- Query enhancement for Australian financial terms
- Context retrieval and response synthesis
- Integration with financial calculations

#### Financial Calculator (`src/models/financial_calculator.py`)
- Australian tax brackets and super guarantee rates
- Emergency fund and investment allocation calculations
- Risk profiling and retirement projections
- Salary sacrifice benefit analysis

#### Data Collectors (`src/data/collectors/`)
- **ABS Collector**: Household income, superannuation statistics
- **ASX Collector**: Stock prices via yfinance API
- **Metals Collector**: Precious metals prices in AUD
- **News Collector**: Financial news RSS feeds

#### Vector Database (`src/data/database/chroma_manager.py`)
- ChromaDB integration for document storage
- Semantic search capabilities
- Metadata filtering and retrieval

## 🧪 Testing & Evaluation

### Walert-Style Evaluation Framework
Following academic research standards:

```bash
# Run full evaluation suite
python src/evaluation/benchmarks.py

# Performance metrics
python -m pytest tests/ --cov=src --cov-report=html
```

**Evaluation Metrics:**
- **NDCG@1, @3, @5** - Retrieval effectiveness
- **ROUGE-1, ROUGE-L** - Response quality
- **BERTScore** - Semantic similarity
- **Unanswered Question Rate** - System reliability
- **Financial Accuracy** - Domain-specific validation

### Test Dataset
- **20 financial topics** (emergency funds, super, investments)
- **30+ knowledge base passages** with Australian context
- **Ground truth relevance judgments** for evaluation
- **Sample user profiles** for testing calculations

## 📊 Sample Queries

The system can answer questions like:

**Emergency Funds:**
- "How much should I save for an emergency fund with $3,000 monthly expenses?"

**Superannuation:**
- "I earn $75,000 annually. How much should I salary sacrifice to super?"

**Investments:**
- "What's a good portfolio allocation for a 35-year-old Australian investor?"

**Precious Metals:**
- "What's the current gold price in Australian dollars and should I invest?"

**ASX Stocks:**
- "Which Australian mining stocks benefit from rising gold prices?"

## 🔧 Configuration

### Environment Variables
```bash
# .env file
PYTHONPATH=./src
CHROMADB_PATH=./data/processed/chroma_db
EMBEDDING_MODEL=FinLang/finance-embeddings-investopedia
LOG_LEVEL=INFO
```

### Configuration Files
- `config/development.yaml` - Development settings
- `config/production.yaml` - Production configuration
- `.vscode/settings.json` - VS Code workspace settings

## 📈 Performance

**System Capabilities:**
- **Response Time**: <3 seconds for typical queries
- **Knowledge Base**: 300+ financial documents
- **Embedding Model**: Finance-optimized transformers
- **Vector Database**: ChromaDB with persistent storage
- **Concurrent Users**: Supports multiple Streamlit sessions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Workflow
```bash
# Setup development environment
python scripts/setup_environment.py

# Run tests before committing
python -m pytest tests/ -v

# Format code
black src/ app/ tests/

# Lint code
flake8 src/ app/ --max-line-length=88
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Walert Project** - RAG evaluation framework methodology
- **Australian Bureau of Statistics** - Household income and economic data
- **APRA** - Superannuation statistics and regulations
- **Reserve Bank of Australia** - Economic indicators and monetary policy
- **ASX** - Market data and investment information

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

## 🗺️ Roadmap

- [ ] **Multi-language Support** - Add support for other languages
- [ ] **Real-time Data Streams** - Live market data integration
- [ ] **Advanced Analytics** - Portfolio performance tracking
- [ ] **Mobile App** - React Native or Flutter interface
- [ ] **API Gateway** - RESTful API for third-party integration
- [ ] **Machine Learning** - Personalized recommendation engine

---

**Built with ❤️ for Australian investors**
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated algorithmic trading system for the Indian stock market that combines automated model-driven trading with manual trading capabilities. The system consists of a **Next.js frontend** and **FastAPI backend** with real-time WebSocket communication, designed for local development and deployment.

### Key Architecture Components
- **Frontend**: Next.js 15+ with TypeScript, Radix UI, Tailwind CSS, and Lightweight Charts
- **Backend**: FastAPI with Python 3.11+, handling live trading, backtesting, and Fyers API integration
- **Trading Engine**: PPO-based reinforcement learning model with TradingEnv for position management
- **Data Pipeline**: Real-time and historical data processing with pandas/numpy
- **Authentication**: JWT-based session management with HTTP-only cookies

## Development Commands

### Frontend Development
```bash
cd frontend
npm run dev          # Start Next.js dev server on port 3000
npm run build        # Build for production  
npm run lint         # Run ESLint
npm run start        # Start production server
```

### Backend Development
```bash
# From root directory
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload  # Start FastAPI server

# Run both frontend and backend concurrently (from frontend/)
npm run dev          # Starts both Next.js and FastAPI using concurrently
```

### Python Environment
```bash
# Activate virtual environment (Windows)
activate_venv.bat

# Install dependencies
pip install -r requirements.txt        # Core dependencies
pip install -r backend/requirements.txt # Backend-specific
pip install -r api/requirements.txt     # Lightweight API deps
```

### Testing
```bash
# Python tests
pytest                    # Run all tests
pytest tests/test_*.py    # Run specific test files
pytest -v                 # Verbose output

# Frontend tests (when implemented)
cd frontend
npm test                  # Run Jest tests
```

### Model Training and Backtesting
```bash
python run_training.py              # Train universal trading model
python run_unified_backtest.py      # Run backtesting
python run_live_bot.py              # Start live trading bot
```

## High-Level Architecture

### Service-Oriented Backend Structure
The backend follows a service-oriented architecture with these key services:

- **LiveTradingService** (`src/trading/live_trading_service.py`): Core trading logic, model integration, WebSocket communication
- **BacktestService** (`src/trading/backtest_service.py`): Historical backtesting with real-time progress updates  
- **FyersClient** (`src/trading/fyers_client.py`): Abstraction layer for Fyers API interactions
- **TradingEnv** (`src/backtesting/environment.py`): Reinforcement learning environment for position management
- **DataLoader** (`src/utils/data_loader.py`): Data processing and feature generation pipeline

### Frontend Architecture
- **App Router**: Next.js 15 App Router pattern with `app/` directory structure
- **Real-time Communication**: WebSocket connections for live data streaming and trading updates
- **State Management**: React Context and custom hooks for global state
- **UI Components**: Radix UI primitives with shadcn/ui patterns and Tailwind CSS styling
- **Charting**: Lightweight Charts library for real-time financial data visualization

### Data Flow
1. **Historical Data**: Fyers API → DataLoader → Feature Processing → Model Training
2. **Live Trading**: Fyers WebSocket → LiveTradingService → Model Inference → Trade Execution
3. **Frontend Updates**: Backend Services → WebSocket → Frontend Real-time Updates
4. **Manual Trading**: Frontend Form → FastAPI Endpoint → LiveTradingService → Fyers API

## Key Configuration Files

### Trading Configuration
- `config/instruments.yaml`: Available trading instruments and timeframes
- `config/training_sequence.yaml`: Model training parameters and architecture
- `models/universal_final_model.pth`: Pre-trained PPO trading model

### API Documentation
- `fyers_docs.txt`: Comprehensive Fyers API documentation and integration details
- `docs/architecture.md`: Detailed system architecture specifications
- `docs/prd.md`: Product requirements with user stories and acceptance criteria

### Deployment Configuration
- `vercel.json`: Vercel deployment configuration for Next.js + Python serverless functions
- `DEPLOYMENT.md`: Step-by-step deployment guide for Vercel
- `api/index.py`: Lightweight serverless API entry point for production

## Important Implementation Notes

### Authentication Flow
- Uses Fyers OAuth 2.0 with TOTP for initial authentication
- JWT tokens stored in HTTP-only cookies for session management  
- All protected routes require valid JWT token via `get_current_user` dependency

### Real-time Data Handling
- WebSocket connections managed per user session
- Live data streaming from Fyers API through LiveTradingService
- Frontend charts update in real-time with tick-by-tick data

### Trading Model Integration
- Universal PPO model (`universal_final_model.pth`) generates trading signals
- TradingEnv handles position management, stop-loss, and target calculations
- Capital-aware quantity validation before trade execution

### Error Handling and Logging
- Comprehensive logging throughout backend services
- WebSocket reconnection logic for stable real-time connections
- Graceful fallbacks when model loading fails (uses random actions)

### Testing Strategy  
- Unit tests for individual components in `tests/` directory
- Integration tests for full trading pipeline
- Use `pytest` for Python testing with fixtures in `conftest.py`

## Development Workflow

1. **New Feature Development**: Follow the epic/story structure in `docs/prd.md`
2. **Code Organization**: Maintain service separation and use existing patterns
3. **Real-time Features**: Use WebSocket broadcasting pattern from LiveTradingService
4. **API Development**: Follow FastAPI patterns with proper error handling and logging
5. **Frontend Integration**: Use established component patterns with proper TypeScript typing

## Production Considerations

- All API keys and secrets managed through environment variables
- Model files and data stored locally (not in cloud storage)
- JSON-based trade logging for performance analysis
- CORS configured for Vercel deployment domains
- Serverless deployment ready with lightweight API subset
#!/bin/bash
# =============================================================================
# Historical QuantArena - Full System Launcher
# =============================================================================
#
# This script runs the complete Historical QuantArena system:
# 1. Runs a historical simulation with multi-agent trading
# 2. Starts the FastAPI backend server
# 3. Starts the Streamlit frontend UI
#
# Usage:
#   ./scripts/run_historical_quant_arena.sh
#   ./scripts/run_historical_quant_arena.sh --symbols SPY,QQQ --start 2020-01-01
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
# Learning showcase: SPY with MTF SuperTrader system from earliest data
SYMBOLS="${SYMBOLS:-SPY}"
START_DATE="${START_DATE:-2000-01-03}"
END_DATE="${END_DATE:-2025-12-09}"
EQUITY="${EQUITY:-100000}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
SKIP_SIMULATION="${SKIP_SIMULATION:-false}"
ENABLE_LEARNING="${ENABLE_LEARNING:-true}"
ENABLE_MTF="${ENABLE_MTF:-true}"
EXEC_TIMEFRAME="${EXEC_TIMEFRAME:-daily}"
USE_SUPER_TRADER="${USE_SUPER_TRADER:-true}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        --start)
            START_DATE="$2"
            shift 2
            ;;
        --end)
            END_DATE="$2"
            shift 2
            ;;
        --equity)
            EQUITY="$2"
            shift 2
            ;;
        --backend-port)
            BACKEND_PORT="$2"
            shift 2
            ;;
        --frontend-port)
            FRONTEND_PORT="$2"
            shift 2
            ;;
        --skip-simulation)
            SKIP_SIMULATION=true
            shift
            ;;
        --enable-learning)
            ENABLE_LEARNING="$2"
            shift 2
            ;;
        --help|-h)
            echo "Historical QuantArena - Full System Launcher"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --symbols SYMS      Comma-separated symbols (default: SPY,QQQ)"
            echo "  --start DATE        Start date YYYY-MM-DD (default: 2024-01-01)"
            echo "  --end DATE          End date YYYY-MM-DD (default: 2024-06-01)"
            echo "  --equity AMOUNT     Initial equity (default: 100000)"
            echo "  --backend-port PORT Backend API port (default: 8000)"
            echo "  --frontend-port PORT Frontend UI port (default: 8501)"
            echo "  --skip-simulation   Skip simulation, just start UI servers"
            echo "  --enable-learning   Enable learning system (default: true)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --symbols SPY,QQQ,IWM --start 2020-01-01 --end 2024-01-01"
            echo "  $0 --skip-simulation  # Just start UI with existing data"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║     █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗                    ║"
echo "║    ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗                   ║"
echo "║    ███████║██║     ██████╔╝███████║███████║                   ║"
echo "║    ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║                   ║"
echo "║    ██║  ██║███████╗██║     ██║  ██║██║  ██║                   ║"
echo "║    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝                   ║"
echo "║                                                               ║"
echo "║    █████╗ ██████╗ ███████╗███╗   ██╗ █████╗                   ║"
echo "║   ██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗                  ║"
echo "║   ███████║██████╔╝█████╗  ██╔██╗ ██║███████║                  ║"
echo "║   ██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██╔══██║                  ║"
echo "║   ██║  ██║██║  ██║███████╗██║ ╚████║██║  ██║                  ║"
echo "║   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝                  ║"
echo "║                                                               ║"
echo "║       LLM-Powered Multi-Agent Trading Simulation              ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Change to project root
cd "$PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT/packages:$PYTHONPATH"

# Load .env file for OPENAI_API_KEY
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
    echo -e "${GREEN}  ✓ Loaded .env file${NC}"
fi

# Check for OpenAI API key
if [[ -n "$OPENAI_API_KEY" ]]; then
    echo -e "${GREEN}  ✓ OpenAI API key detected - LLM agents enabled${NC}"
else
    echo -e "${RED}  ⚠ Warning: OPENAI_API_KEY not found!${NC}"
    echo -e "${RED}    Add OPENAI_API_KEY=sk-... to .env file${NC}"
fi

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    
    # Kill background processes
    if [[ -n "$BACKEND_PID" ]]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [[ -n "$FRONTEND_PID" ]]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    
    # Also kill any processes on our ports
    lsof -ti:$BACKEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:$FRONTEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    
    echo -e "${GREEN}Services stopped.${NC}"
}

trap cleanup EXIT

# =============================================================================
# Step 1: Prepare database copy (to avoid MCP server lock conflicts)
# =============================================================================
echo -e "${BLUE}[1/4] Preparing database...${NC}"

DB_SOURCE="$PROJECT_ROOT/data/trader.duckdb"
DB_SIM="$PROJECT_ROOT/data/trader_sim.duckdb"

if [[ -f "$DB_SOURCE" ]]; then
    cp "$DB_SOURCE" "$DB_SIM"
    echo -e "${GREEN}  ✓ Database copy created${NC}"
else
    echo -e "${YELLOW}  ⚠ Source database not found, simulation will use fallback data${NC}"
fi

# =============================================================================
# Step 2: Run Historical Simulation
# =============================================================================
if [[ "$SKIP_SIMULATION" != "true" ]]; then
    echo -e "${BLUE}[2/4] Running historical simulation...${NC}"
    echo -e "  Symbols: ${SYMBOLS}"
    echo -e "  Period: ${START_DATE} to ${END_DATE}"
    echo -e "  Initial equity: \$${EQUITY}"
    echo ""
    
    # Build command with optional flags
    SIM_CMD="python -m quant_arena.historical.run \
        --symbols $SYMBOLS \
        --start $START_DATE \
        --end $END_DATE \
        --equity $EQUITY \
        --exec-timeframe $EXEC_TIMEFRAME"
    
    if [[ "$ENABLE_LEARNING" == "true" ]]; then
        SIM_CMD="$SIM_CMD --enable-learning"
        echo -e "  Learning system: ${GREEN}ENABLED${NC}"
    fi
    
    if [[ "$ENABLE_MTF" != "true" ]]; then
        SIM_CMD="$SIM_CMD --no-mtf"
        echo -e "  MTF Analysis: ${YELLOW}DISABLED${NC}"
    else
        echo -e "  MTF Analysis: ${GREEN}ENABLED${NC}"
    fi
    
    if [[ "$USE_SUPER_TRADER" == "true" ]]; then
        echo -e "  SuperTrader: ${GREEN}ENABLED${NC}"
    fi
    
    eval $SIM_CMD
    
    echo -e "${GREEN}  ✓ Simulation complete${NC}"
else
    echo -e "${YELLOW}[2/4] Skipping simulation (--skip-simulation flag)${NC}"
fi

# =============================================================================
# Step 3: Start Backend API Server
# =============================================================================
echo -e "${BLUE}[3/4] Starting backend API server...${NC}"

# Kill any existing process on the port
lsof -ti:$BACKEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# Start uvicorn in background
uvicorn examples.historical_quant_arena_ui.backend.api:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --log-level warning &
BACKEND_PID=$!

# Wait for backend to be ready
echo -e "  Waiting for backend to start..."
for i in {1..30}; do
    if curl -s "http://localhost:$BACKEND_PORT/" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Backend API running at http://localhost:$BACKEND_PORT${NC}"
        break
    fi
    sleep 1
done

# =============================================================================
# Step 4: Start Frontend UI
# =============================================================================
echo -e "${BLUE}[4/4] Starting frontend UI...${NC}"

# Kill any existing process on the port
lsof -ti:$FRONTEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# Start streamlit in background
streamlit run examples/historical_quant_arena_ui/frontend/app.py \
    --server.port "$FRONTEND_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!

# Wait for frontend to be ready
echo -e "  Waiting for frontend to start..."
for i in {1..30}; do
    if curl -s "http://localhost:$FRONTEND_PORT/" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Frontend UI running at http://localhost:$FRONTEND_PORT${NC}"
        break
    fi
    sleep 1
done

# =============================================================================
# Ready!
# =============================================================================
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    SYSTEM READY                               ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  Frontend UI:  http://localhost:$FRONTEND_PORT                         ║${NC}"
echo -e "${GREEN}║  Backend API:  http://localhost:$BACKEND_PORT                          ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  Press Ctrl+C to stop all services                            ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Open browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    sleep 2
    open "http://localhost:$FRONTEND_PORT" 2>/dev/null || true
fi

# Wait for user interrupt
echo "Services running. Press Ctrl+C to stop..."
wait

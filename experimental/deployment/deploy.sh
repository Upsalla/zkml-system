#!/bin/bash
#
# zkML System Deployment Script
#
# Usage:
#   ./deploy.sh local    # Local installation
#   ./deploy.sh docker   # Docker deployment
#   ./deploy.sh test     # Run tests
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Output colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Local installation
deploy_local() {
    log_info "Starting local installation..."
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment (optional)
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Install dependencies
    log_info "Installing dependencies..."
    pip3 install -r requirements.txt
    
    # Install package
    log_info "Installing zkml-system..."
    pip3 install -e .
    
    log_info "Installation complete!"
    log_info "Available commands:"
    echo "  zkml --help           # CLI-Hilfe"
    echo "  zkml serve            # API-Server starten"
    echo "  zkml init             # Beispiel-Netzwerk erstellen"
    echo "  zkml prove -n ... -i ...  # Proof generieren"
}

# Docker deployment
deploy_docker() {
    log_info "Starting Docker deployment..."
    
    cd "$PROJECT_DIR"
    
    # Build Docker image
    log_info "Building Docker image..."
    docker build -t zkml-system:latest -f deployment/docker/Dockerfile .
    
    # Start container
    log_info "Starting container..."
    docker-compose -f deployment/docker/docker-compose.yml up -d
    
    # Wait for health check
    log_info "Waiting for server..."
    sleep 5
    
    # Health-Check
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        log_info "Server is running!"
        log_info "API available at: http://localhost:8000"
        log_info "Documentation: http://localhost:8000/docs"
    else
        log_error "Server unreachable!"
        docker-compose -f deployment/docker/docker-compose.yml logs
        exit 1
    fi
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cd "$PROJECT_DIR"
    
    # Integration tests
    log_info "Integration tests..."
    python3 -u plonk/test_integration.py
    
    # CLI-Test
    log_info "CLI-Test..."
    python3 -m deployment.cli.main init --output /tmp/test_network.json
    python3 -m deployment.cli.main compile -n /tmp/test_network.json -i /tmp/test_network_input.json
    
    log_info "All tests passed!"
}

# API test
test_api() {
    log_info "Testing API..."
    
    # Health-Check
    curl -s http://localhost:8000/health | python3 -m json.tool
    
    # Compile test
    log_info "Compile endpoint..."
    curl -s -X POST http://localhost:8000/compile \
        -H "Content-Type: application/json" \
        -d '{
            "network": {
                "name": "test",
                "input_size": 4,
                "layers": [
                    {
                        "type": "dense",
                        "weights": [[1,2,3,4], [5,6,7,8]],
                        "biases": [1, 2],
                        "activation": "gelu"
                    }
                ]
            },
            "inputs": [10, 20, 30, 40]
        }' | python3 -m json.tool
    
    log_info "API tests complete!"
}

# Help
show_help() {
    echo "zkML System Deployment Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  local     Local installation (pip install)"
    echo "  docker    Docker deployment (docker-compose)"
    echo "  test      Run tests"
    echo "  api-test  Test API endpoints"
    echo "  help      Show this help"
}

# Main
case "${1:-help}" in
    local)
        deploy_local
        ;;
    docker)
        deploy_docker
        ;;
    test)
        run_tests
        ;;
    api-test)
        test_api
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

#!/bin/bash
#
# zkML System Deployment Script
#
# Usage:
#   ./deploy.sh local    # Lokale Installation
#   ./deploy.sh docker   # Docker-Deployment
#   ./deploy.sh test     # Tests ausführen
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Farben für Output
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

# Lokale Installation
deploy_local() {
    log_info "Starte lokale Installation..."
    
    cd "$PROJECT_DIR"
    
    # Virtuelle Umgebung erstellen (optional)
    if [ ! -d "venv" ]; then
        log_info "Erstelle virtuelle Umgebung..."
        python3 -m venv venv
    fi
    
    # Dependencies installieren
    log_info "Installiere Dependencies..."
    pip3 install -r requirements.txt
    
    # Package installieren
    log_info "Installiere zkml-system..."
    pip3 install -e .
    
    log_info "Installation abgeschlossen!"
    log_info "Verfügbare Befehle:"
    echo "  zkml --help           # CLI-Hilfe"
    echo "  zkml serve            # API-Server starten"
    echo "  zkml init             # Beispiel-Netzwerk erstellen"
    echo "  zkml prove -n ... -i ...  # Proof generieren"
}

# Docker-Deployment
deploy_docker() {
    log_info "Starte Docker-Deployment..."
    
    cd "$PROJECT_DIR"
    
    # Docker-Image bauen
    log_info "Baue Docker-Image..."
    docker build -t zkml-system:latest -f deployment/docker/Dockerfile .
    
    # Container starten
    log_info "Starte Container..."
    docker-compose -f deployment/docker/docker-compose.yml up -d
    
    # Warte auf Health-Check
    log_info "Warte auf Server..."
    sleep 5
    
    # Health-Check
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        log_info "Server läuft!"
        log_info "API verfügbar unter: http://localhost:8000"
        log_info "Dokumentation: http://localhost:8000/docs"
    else
        log_error "Server nicht erreichbar!"
        docker-compose -f deployment/docker/docker-compose.yml logs
        exit 1
    fi
}

# Tests ausführen
run_tests() {
    log_info "Führe Tests aus..."
    
    cd "$PROJECT_DIR"
    
    # Integrationstests
    log_info "Integrationstests..."
    python3 -u plonk/test_integration.py
    
    # CLI-Test
    log_info "CLI-Test..."
    python3 -m deployment.cli.main init --output /tmp/test_network.json
    python3 -m deployment.cli.main compile -n /tmp/test_network.json -i /tmp/test_network_input.json
    
    log_info "Alle Tests bestanden!"
}

# API-Test
test_api() {
    log_info "Teste API..."
    
    # Health-Check
    curl -s http://localhost:8000/health | python3 -m json.tool
    
    # Compile-Test
    log_info "Compile-Endpunkt..."
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
    
    log_info "API-Tests abgeschlossen!"
}

# Hilfe
show_help() {
    echo "zkML System Deployment Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  local     Lokale Installation (pip install)"
    echo "  docker    Docker-Deployment (docker-compose)"
    echo "  test      Tests ausführen"
    echo "  api-test  API-Endpunkte testen"
    echo "  help      Diese Hilfe anzeigen"
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
        log_error "Unbekannter Befehl: $1"
        show_help
        exit 1
        ;;
esac

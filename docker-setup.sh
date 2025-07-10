#!/bin/bash

# TradingAgents Docker Setup Script
# This script helps you set up and run TradingAgents using Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check environment variables
check_env_vars() {
    print_info "Checking required environment variables..."
    
    if [ -z "$OPENAI_API_KEY" ]; then
        print_error "OPENAI_API_KEY is not set"
        echo "Please set it: export OPENAI_API_KEY='your_key_here'"
        return 1
    fi
    
    if [ -z "$FINNHUB_API_KEY" ]; then
        print_error "FINNHUB_API_KEY is not set"
        echo "Please set it: export FINNHUB_API_KEY='your_key_here'"
        return 1
    fi
    
    print_success "Required environment variables are set"
    
    # Check optional variables
    if [ -z "$GOOGLE_API_KEY" ]; then
        print_warning "GOOGLE_API_KEY is not set (optional for Gemini models)"
    fi
    
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        print_warning "ANTHROPIC_API_KEY is not set (optional for Claude models)"
    fi
    
    return 0
}

# Build the Docker image
build_image() {
    print_info "Building Docker image..."
    docker-compose build
    print_success "Docker image built successfully"
}

# Show usage
show_usage() {
    echo "TradingAgents Docker Setup"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build     Build the Docker image"
    echo "  run       Run the application (CLI interface)"
    echo "  example   Run the example script (main.py)"
    echo "  shell     Start an interactive shell in the container"
    echo "  logs      Show container logs"
    echo "  stop      Stop all containers"
    echo "  clean     Remove containers and images"
    echo "  help      Show this help message"
    echo ""
    echo "Prerequisites:"
    echo "  - Set OPENAI_API_KEY environment variable"
    echo "  - Set FINNHUB_API_KEY environment variable"
    echo "  - Optional: GOOGLE_API_KEY, ANTHROPIC_API_KEY"
    echo ""
    echo "Example:"
    echo "  export OPENAI_API_KEY='your_key'"
    echo "  export FINNHUB_API_KEY='your_key'"
    echo "  $0 build && $0 run"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_docker
        build_image
        ;;
    "run")
        check_docker
        if ! check_env_vars; then
            exit 1
        fi
        print_info "Starting TradingAgents CLI..."
        docker-compose run --rm tradingagents python -m cli.main
        ;;
    "example")
        check_docker
        if ! check_env_vars; then
            exit 1
        fi
        print_info "Running example script..."
        docker-compose run --rm tradingagents python main.py
        ;;
    "shell")
        check_docker
        print_info "Starting interactive shell..."
        docker-compose run --rm tradingagents bash
        ;;
    "logs")
        check_docker
        docker-compose logs -f tradingagents
        ;;
    "stop")
        check_docker
        print_info "Stopping containers..."
        docker-compose down
        print_success "Containers stopped"
        ;;
    "clean")
        check_docker
        print_warning "This will remove all containers, volumes, and images related to TradingAgents"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            docker rmi tradingagents:latest 2>/dev/null || true
            print_success "Cleanup completed"
        else
            print_info "Cleanup cancelled"
        fi
        ;;
    "help"|*)
        show_usage
        ;;
esac 
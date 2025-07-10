# TradingAgents Docker Setup

This document provides instructions for running TradingAgents using Docker.

## Quick start

```
docker build -t tradingagents:latest .
docker-compose run --rm tradingagents bash
docker-compose run --rm tradingagents python simple_test_backtest.py
```

Then inside the container, you can run:
`python main.py`
or
`python -m cli.main  # for the CLI interface`

## Prerequisites

- Docker installed on your system
- Docker Compose (usually comes with Docker Desktop)
- Required API keys (see below)

## Required API Keys

Before running the container, you need to obtain the following API keys:

1. **OpenAI API Key**: Required for LLM functionality
   - Get it from: https://platform.openai.com/api-keys
   
2. **FinnHub API Key**: Required for financial data
   - Get it from: https://finnhub.io/register (free tier available)

3. **Optional API Keys**:
   - Google API Key (for Gemini models)
   - Anthropic API Key (for Claude models)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export FINNHUB_API_KEY="your_finnhub_api_key_here"
   
   # Optional
   export GOOGLE_API_KEY="your_google_api_key_here"
   export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
   ```

2. **Build and run the container**:
   ```bash
   docker-compose up --build
   ```

3. **To run in detached mode**:
   ```bash
   docker-compose up -d --build
   ```

4. **To stop the container**:
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker directly

1. **Build the image**:
   ```bash
   docker build -t tradingagents:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -it \
     -e OPENAI_API_KEY="your_openai_api_key_here" \
     -e FINNHUB_API_KEY="your_finnhub_api_key_here" \
     -p 8000:8000 \
     tradingagents:latest
   ```

## Running Different Commands

### Run the CLI Interface
```bash
docker-compose run --rm tradingagents python -m cli.main
```

### Run the Python Example
```bash
docker-compose run --rm tradingagents python main.py
```

### Run Custom Python Script
```bash
docker-compose run --rm tradingagents python your_script.py
```

### Interactive Shell
```bash
docker-compose run --rm tradingagents bash
```

## Development Mode

If you want to make changes to the code and see them reflected immediately:

1. **Uncomment the volume mount in docker-compose.yml**:
   ```yaml
   volumes:
     - .:/app  # Uncomment this line
   ```

2. **Rebuild and run**:
   ```bash
   docker-compose up --build
   ```

## Troubleshooting

### Common Issues

1. **API Key Not Set**:
   - Error: Missing API keys
   - Solution: Make sure you've exported the required environment variables

2. **Permission Errors**:
   - Error: Permission denied when accessing files
   - Solution: The container runs as a non-root user for security. If you encounter permission issues, you can modify the Dockerfile to run as root (not recommended for production)

3. **Port Already in Use**:
   - Error: Port 8000 is already allocated
   - Solution: Change the port mapping in docker-compose.yml or stop the service using that port

4. **Build Failures**:
   - Error: Package installation failures
   - Solution: Try rebuilding without cache: `docker-compose build --no-cache`

### Useful Docker Commands

```bash
# View running containers
docker ps

# View all containers
docker ps -a

# View logs
docker-compose logs tradingagents

# Follow logs in real-time
docker-compose logs -f tradingagents

# Remove all containers and volumes
docker-compose down -v

# Remove images
docker rmi tradingagents:latest

# Clean up unused Docker resources
docker system prune -a
```

## Configuration

The container includes Redis for caching. You can modify the `docker-compose.yml` file to:

- Change port mappings
- Add additional environment variables
- Mount additional volumes
- Configure Redis settings

## Security Notes

- The container runs as a non-root user for security
- API keys are passed as environment variables (never hardcode them)
- Consider using Docker secrets for production deployments
- The container exposes port 8000 for potential web interfaces

## Production Deployment

For production deployment, consider:

1. Using a proper secrets management system
2. Setting up proper logging and monitoring
3. Using a production-grade database instead of Redis
4. Implementing proper backup strategies
5. Using multi-stage builds for smaller images
6. Setting up health checks

## Support

For issues related to Docker setup, please check:
1. Docker and Docker Compose are properly installed
2. All required API keys are set
3. No port conflicts exist
4. Sufficient disk space is available

For application-specific issues, refer to the main README.md file. 
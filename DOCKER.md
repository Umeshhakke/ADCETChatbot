# 🐳 Docker Deployment Guide

## Quick Start

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f chatbot

# Stop the application
docker-compose down
```

## Building the Image

```bash
docker build -t college-chatbot:latest .
```

## Running the Container

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Using Docker CLI

```bash
docker run -d \
  --name college-chatbot \
  -p 8001:8001 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env \
  --restart unless-stopped \
  college-chatbot:latest
```

## Container Management

### View Logs
```bash
# All logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100 chatbot

# Docker CLI
docker logs -f college-chatbot
```

### Check Status
```bash
docker-compose ps

# Or
docker ps | grep college-chatbot
```

### Restart Container
```bash
docker-compose restart

# Or
docker restart college-chatbot
```

### Stop and Remove
```bash
docker-compose down

# Or
docker stop college-chatbot
docker rm college-chatbot
```

## Health Check

The container includes a health check that runs every 30 seconds:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' college-chatbot

# View health check logs
docker inspect --format='{{json .State.Health}}' college-chatbot | jq
```

## Volume Mounts

- `./logs:/app/logs` - Application logs
- `./.env:/app/.env` - Environment configuration

## Environment Variables

Override in `docker-compose.yml` or pass via `-e`:

```bash
docker run -d \
  -e HOST=0.0.0.0 \
  -e PORT=8001 \
  -e RATE_LIMIT=20 \
  -e RATE_WINDOW=60 \
  college-chatbot:latest
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs chatbot

# Check if port is already in use
lsof -i :8001
```

### Permission issues with logs
```bash
# Fix permissions
chmod -R 755 logs/
```

### Rebuild after code changes
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Production Considerations

1. **Use specific image tags** instead of `latest`
2. **Set resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '1'
         memory: 512M
   ```
3. **Use Docker secrets** for sensitive data
4. **Enable log rotation** to prevent disk space issues
5. **Use a reverse proxy** (nginx/traefik) for SSL termination

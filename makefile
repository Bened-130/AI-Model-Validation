# AI Validation Framework - Docker Commands

.PHONY: help build up down logs test clean

help:
	@echo "AI Validation Framework - Available Commands:"
	@echo "  make build      - Build all Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs from all services"
	@echo "  make api        - Start only API service"
	@echo "  make demo       - Start Streamlit demo"
	@echo "  make mlflow     - Start MLflow tracking server"
	@echo "  make dev        - Start development stack (includes Jupyter)"
	@echo "  make test       - Run tests in Docker"
	@echo "  make clean      - Remove all containers and volumes"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

api:
	docker-compose up -d api

demo:
	docker-compose up -d streamlit mlflow

mlflow:
	docker-compose up -d mlflow

dev:
	docker-compose --profile dev up -d

test:
	docker-compose run --rm api pytest tests/ -v

clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

shell-api:
	docker-compose exec api /bin/bash

shell-mlflow:
	docker-compose exec mlflow /bin/bash
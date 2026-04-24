.PHONY: bootstrap up down reset logs ps test lint mypy build smoke

bootstrap:
	./scripts/bootstrap.sh

up:
	docker compose up -d postgres redis
	docker compose --profile ops run --rm migrate
	docker compose up -d api worker

down:
	docker compose down

reset:
	docker compose down -v
	./scripts/bootstrap.sh

logs:
	docker compose logs -f api worker

ps:
	docker compose ps

smoke:
	./scripts/smoke_compose.sh

test:
	python -m pytest -q

lint:
	python -m ruff check .

mypy:
	uv run --extra dev mypy src

build:
	uv build

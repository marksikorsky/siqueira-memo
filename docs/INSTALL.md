# One-command install

Siqueira Memo ships with a Docker Compose bootstrap so operators do **not** need
to manually install Postgres, pgvector, Redis, Python packages, or migrations.

## Requirements

- Docker Engine / Docker Desktop
- Docker Compose v2 (`docker compose version`)
- `curl`
- `python3`

## Start everything

```bash
git clone https://github.com/<owner>/siqueira-memo.git
cd siqueira-memo
./scripts/bootstrap.sh
```

The script will:

1. generate `.env` with local secrets if it does not exist;
2. pull `pgvector/pgvector:pg16` and `redis:7-alpine`;
3. build the Siqueira image;
4. start Postgres + Redis;
5. run `alembic upgrade head`;
6. start API + worker;
7. verify `/healthz` and `/readyz`;
8. if Hermes is present, install the `siqueira-memo` MemoryProvider, write
   `SIQUEIRA_*` settings into `~/.hermes/.env`, set
   `memory.provider: siqueira-memo` in `~/.hermes/config.yaml`, verify Hermes
   provider discovery, and restart the Hermes gateway if it is running.

No real LLM or embedding API key is required for the default local install:
`SIQUEIRA_EMBEDDING_PROVIDER=mock` is used.

## Options

```bash
# Start only the Docker service; do not touch Hermes.
SIQUEIRA_INSTALL_HERMES_PROVIDER=false ./scripts/bootstrap.sh

# Force Hermes install/restart failures to be fatal in the provider installer.
SIQUEIRA_RESTART_HERMES_GATEWAY=force ./scripts/install_hermes_provider.sh

# Install provider but do not restart gateway.
SIQUEIRA_RESTART_HERMES_GATEWAY=false ./scripts/install_hermes_provider.sh
```

## Useful commands

```bash
docker compose ps
docker compose logs -f api worker
./scripts/smoke_compose.sh
docker compose down
```

Destructive reset:

```bash
docker compose down -v
./scripts/bootstrap.sh
```

## Ports

Defaults are bound to localhost only:

| service | default |
|---|---:|
| API | `127.0.0.1:8787` |
| Postgres | `127.0.0.1:5432` |
| Redis | `127.0.0.1:6379` |

Override in `.env`:

```dotenv
SIQUEIRA_PUBLIC_PORT=8788
SIQUEIRA_POSTGRES_PORT=15432
SIQUEIRA_REDIS_PORT=16379
```

## Production notes

For real deployment:

- set `SIQUEIRA_ENV=production`;
- replace generated local secrets;
- use real embedding provider credentials if needed;
- put the API behind TLS/reverse proxy;
- decide whether native monthly table partitioning is required before enabling
  strict production readiness checks.

The development bootstrap intentionally avoids forcing native Postgres
partition conversion. That keeps first install painless while preserving the
production code paths for operators who need them.

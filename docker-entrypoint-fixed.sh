#!/bin/bash
set -e

if [ ! -f /app/morphik.toml ]; then
    cp /app/morphik.toml.default /app/morphik.toml
fi

check_postgres() {
    if [ -n "$POSTGRES_URI" ]; then
        echo "Waiting for PostgreSQL..."
        max_retries=30
        retries=0
        
        # Extract hostname from POSTGRES_URI (format: postgresql+asyncpg://user:pass@host:port/db)
        DB_HOST=$(echo "$POSTGRES_URI" | sed -n 's|.*@\([^:]*\):.*|\1|p')
        DB_PORT=$(echo "$POSTGRES_URI" | sed -n 's|.*:\([0-9]*\)/.*|\1|p')
        DB_USER=$(echo "$POSTGRES_URI" | sed -n 's|.*://\([^:]*\):.*|\1|p')
        DB_NAME=$(echo "$POSTGRES_URI" | sed -n 's|.*/\([^?]*\).*|\1|p')
        
        # Fallback to environment variables if extraction fails
        DB_HOST=${DB_HOST:-${DB_HOST:-postgres}}
        DB_PORT=${DB_PORT:-5432}
        DB_USER=${DB_USER:-${PGUSER:-postgres}}
        DB_NAME=${DB_NAME:-${DB_NAME:-private_ai_dev}}
        
        echo "Connecting to PostgreSQL at $DB_HOST:$DB_PORT as user $DB_USER to database $DB_NAME"
        
        until PGPASSWORD=$PGPASSWORD pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER"; do
            retries=$((retries + 1))
            if [ $retries -eq $max_retries ]; then
                echo "Error: PostgreSQL did not become ready in time"
                exit 1
            fi
            echo "Waiting for PostgreSQL... (Attempt $retries/$max_retries)"
            sleep 2
        done
        echo "PostgreSQL is ready!"
        
        if ! PGPASSWORD=$PGPASSWORD psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" > /dev/null 2>&1; then
            echo "Error: Could not connect to PostgreSQL database $DB_NAME with user $DB_USER"
            exit 1
        fi
        echo "PostgreSQL connection verified!"
    fi
}

check_postgres

if [ $# -gt 0 ]; then
    exec "$@"
else
    exec uv run uvicorn core.api:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --loop asyncio --http auto --ws auto --lifespan auto
fi 

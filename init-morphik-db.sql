-- Initialize Morphik database during PostgreSQL startup
-- This script runs automatically in docker-entrypoint-initdb.d

-- Create morphik user with password
CREATE USER morphik WITH PASSWORD 'morphik_password';

-- Create morphik_dev database and assign owner
CREATE DATABASE morphik_dev OWNER morphik;

-- Grant all privileges on morphik_dev to morphik user
GRANT ALL PRIVILEGES ON DATABASE morphik_dev TO morphik;

-- Connect to morphik_dev and enable extensions
\c morphik_dev;

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant schema privileges to morphik user
GRANT ALL ON SCHEMA public TO morphik;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO morphik;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO morphik;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO morphik;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO morphik;

-- Connect back to postgres database
\c postgres;

-- Ensure main database also has vector extension for compatibility
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant connect privileges to postgres database for morphik user (if needed)
GRANT CONNECT ON DATABASE postgres TO morphik; 

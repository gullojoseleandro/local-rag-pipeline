# API service

The service exposes a FastAPI endpoint at /health for health checks. It returns HTTP 200 when the process is ready and HTTP 503 when the database connection is unavailable. The API connects to PostgreSQL for durable order records. The connection pool is limited to 20 concurrent connections per application instance. Increasing the pool without checking the database limit can exhaust PostgreSQL connections.

Requests for order summaries use Redis as a short-lived cache. The cache time-to-live is 60 seconds. Redis is never the source of truth for orders; a cache miss reads PostgreSQL and then repopulates Redis. Writes update PostgreSQL first and invalidate the matching Redis key only after the transaction commits.

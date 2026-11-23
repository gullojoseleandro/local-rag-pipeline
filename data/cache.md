# Cache policy

Redis caches order summaries for 60 seconds. A cache hit can avoid a PostgreSQL read, but every order write still goes to PostgreSQL. The service invalidates a cached summary only after the database transaction succeeds. If Redis is unavailable, the API reads PostgreSQL directly and logs a cache warning; order data remains available while the database is healthy.

The cache is intentionally optional. Redis failure can increase database load, so operators monitor PostgreSQL connection pressure during a cache outage. Cache entries contain derived summaries rather than authentication secrets.

# Operations

The deployment runs two API instances behind a reverse proxy. Operators inspect /health before directing traffic to a new instance. A failed database check makes the endpoint return HTTP 503, which keeps the instance out of rotation. Application logs include a request ID so incidents can be correlated without logging credentials or customer data.

When PostgreSQL reports too many connections, operators check active connections, the API pool size, and the number of running API instances. The documented remediation is to reduce application concurrency or adjust the database capacity after measuring load. Restarting Redis does not resolve a PostgreSQL connection limit.

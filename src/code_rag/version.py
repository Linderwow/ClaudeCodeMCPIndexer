__version__ = "0.1.0"
# Bump when index schema or embedder contract changes. Stamped into index metadata;
# a mismatch on open forces a full reindex rather than silently serving stale vectors.
INDEX_SCHEMA_VERSION = 1

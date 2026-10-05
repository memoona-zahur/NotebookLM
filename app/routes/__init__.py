"""Routers, grouped by what they act on rather than by HTTP verb.

Each module owns one resource: its URLs, and the translation between HTTP and
the domain functions. Grouping by resource means you can answer "who is allowed
to delete a source?" by reading one file, instead of grepping a single flat list
of handlers.
"""
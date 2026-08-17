"""Live-model side: adapters, capture, and artifact writing.

Importing this subpackage does not import a model backend. The backends are
imported inside the adapter factories, so `inspect` and `render` can resolve a
capture spec, estimate memory and render a job without initialising CUDA.
"""

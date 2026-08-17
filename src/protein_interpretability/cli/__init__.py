"""Command line for the things this project does repeatedly.

The plan sketches `collect / predict / intervene / analyze` verbs. Those describe
work that is still moving, and a CLI written ahead of the code it wraps ends up
being changed for reasons that have nothing to do with users. What is already
stable, load-bearing and done by hand every time is narrower:

    reproduce   re-run an archived result from the argv it recorded
    verify      diff two results, with the known non-determinism bands applied
    inspect     static checks on a script before a job is submitted for it

All three are login-node safe: none imports a model backend, and `inspect`
exists precisely to prove that a script would not either.

`reproduce` matters most. Every result file here carries the exact command that
produced it, which makes regeneration mechanical -- but it has been done with a
throwaway script each time, and a throwaway script is where the tolerances and
the known-unstable list get lost.
"""

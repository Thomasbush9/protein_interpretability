"""Can mosaic's Boltz2 wrapper be told not to subsample the MSA?

The adapter comparison showed the wrapper is built with `subsample_msa=True,
num_subsampled_msa=1024` while `pi_core` deliberately sets it False. Whether
that is fixable through the constructor or has to be patched afterwards decides
how the pi_models adapter is configured, so it is worth one job to find out
rather than guessing from the call site.

Prints only; writes nothing.

    sbatch checkout.sbatch probe_mosaic_msa.py
"""

import inspect


def main():
    from mosaic.models.boltz2 import Boltz2

    print("=== Boltz2.__init__ signature")
    try:
        print(inspect.signature(Boltz2.__init__))
    except (TypeError, ValueError) as exc:
        print("unavailable:", exc)

    for name in ("__init__", "load", "from_pretrained"):
        fn = getattr(Boltz2, name, None)
        if fn is None or name == "__init__":
            continue
        try:
            print(f"=== Boltz2.{name} signature\n{inspect.signature(fn)}")
        except (TypeError, ValueError):
            pass

    print("\n=== dataclass fields, if it is one")
    print(getattr(Boltz2, "__dataclass_fields__", {}).keys())

    print("\n=== where the flag actually lives on a built wrapper")
    w = Boltz2()
    mm = getattr(getattr(w, "model", None), "msa_module", None)
    print("wrapper.model.msa_module:", type(mm).__name__ if mm else None)
    for attr in ("subsample_msa", "num_subsampled_msa"):
        print(f"  {attr} = {getattr(mm, attr, '<absent>')}")

    print("\n=== is it writable after construction?")
    if mm is not None and hasattr(mm, "subsample_msa"):
        try:
            import equinox as eqx
            print("  module is an equinox Module:", isinstance(mm, eqx.Module))
            print("  (equinox Modules are frozen; use eqx.tree_at to replace "
                  "the field, or pass the arg at construction)")
        except Exception as exc:
            print("  ", repr(exc))

    print("\n=== does the source mention the argument")
    try:
        src = inspect.getsource(Boltz2)
        for line in src.splitlines():
            if "subsample" in line or "msa_args" in line or "MSAModuleArgs" in line:
                print("   ", line.strip())
    except OSError as exc:
        print("   source unavailable:", exc)


if __name__ == "__main__":
    main()

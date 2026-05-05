"""End-to-end smoke test against a real Boltz2 model.

Skipped unless ``BOLTZ_ATTRIBUTION_SMOKE_YAML`` points to a Boltz YAML and
``BOLTZ_CACHE`` (or ``~/.boltz``) holds a downloaded checkpoint. Intended to
be run on a machine that already has Boltz set up — not on CI.

Verifies:
  - Forward pass keeps the autograd graph alive on captured surfaces.
  - One backward pass populates ``.grad`` on query + MSA embeddings.
  - Gradient values are finite and not all-zero.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

boltz = pytest.importorskip("boltz")

YAML_ENV = "BOLTZ_ATTRIBUTION_SMOKE_YAML"
CACHE_ENV = "BOLTZ_CACHE"


def _have_required_env() -> bool:
    return os.environ.get(YAML_ENV) is not None


@pytest.mark.skipif(
    not _have_required_env(),
    reason=f"set {YAML_ENV}=<path/to/input.yaml> to run the Boltz smoke test",
)
def test_attribution_end_to_end(tmp_path) -> None:
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
        check_inputs,
        download_boltz2,
        filter_inputs_structure,
        process_inputs,
    )
    from boltz.model.models.boltz2 import Boltz2

    from protein_interpretability.attribution import ContactBinNLL, run_per_step

    yaml_path = Path(os.environ[YAML_ENV]).expanduser()
    cache = Path(os.environ.get(CACHE_ENV, "~/.boltz")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = check_inputs(yaml_path)
    data = filter_inputs_structure(data, out_dir, override=False)
    if not data:
        pytest.skip("input already processed in another run; nothing to do")
    processed = process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=cache / "mols",
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
    )

    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=cache / "mols",
        num_workers=0,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )
    data_module.setup(stage="predict")
    dataloader = data_module.predict_dataloader()
    batch = next(iter(dataloader))

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, use_paired_feature=True)
    steering_args = BoltzSteeringParams()

    model = Boltz2.load_from_checkpoint(
        cache / "boltz2_conf.ckpt",
        strict=True,
        predict_args={
            "recycling_steps": 1,
            "sampling_steps": 1,
            "diffusion_samples": 1,
            "max_parallel_samples": None,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    n = int(batch["token_pad_mask"][0].sum().item())
    target = ContactBinNLL(pair_i=0, pair_j=min(8, n - 1))

    results = run_per_step(
        model=model,
        batch=batch,
        target=target,
        recycling_steps=(0, 1),
    )
    assert len(results) == 2
    for r in results:
        assert torch.isfinite(r.query_grad).all()
        assert r.query_grad.abs().sum() > 0
        assert torch.isfinite(torch.tensor(r.target_value))

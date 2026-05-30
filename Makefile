.PHONY: smoke ablation test

smoke:
	bash scripts/run_smoke.sh

ablation:
	bash scripts/run_ablation_synthetic.sh

test:
	PYTHONPATH=src pytest -q

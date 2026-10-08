.PHONY: build api web science science-test capability-validate

build:
	cargo build --workspace

api:
	cargo run -p medai-api

web:
	cd web && npm run dev

science:
	cd science && python3 -m pip install -e .

science-test:
	PYTHONPATH=science/src python3 -m unittest discover -s science/tests -v

capability-validate:
	@for package in integrations/agentmesh/capabilities/*.yaml; do agentmesh capabilities validate "$$package" || exit $$?; done

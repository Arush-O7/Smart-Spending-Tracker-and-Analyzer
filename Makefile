.PHONY: help install dev test lint format app clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies and the package
	pip install -r requirements.txt && pip install -e .

dev:  ## Install development dependencies
	pip install -r requirements-dev.txt && pip install -e .

test:  ## Run the test suite with coverage
	pytest --cov=spending_analyzer --cov-report=term-missing

lint:  ## Check formatting and lint rules
	ruff check .

format:  ## Auto-fix lint issues
	ruff check . --fix

app:  ## Launch the Streamlit dashboard
	streamlit run app.py

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -not -path "./venv/*" -exec rm -rf {} +

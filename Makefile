# Makefile for Coq Proof LLM Benchmark Pipeline
# Provides easy commands for the entire workflow: dataset → proofs → verification → stats

.PHONY: help clean all
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
MODULE_PREFIX := -m src.
VENV_ACTIVATE := source .venv/bin/activate &&

# Default values
LEMMAS_FILE ?= data/lemmas_auto.jsonl
OUTPUT_DIR ?= results/batch
K_SAMPLES ?= 5
MAX_TOKENS ?= 256
TEMPERATURE ?= 0.7
TOP_P ?= 0.9
TIMEOUT ?= 15
LIMIT ?= 0

# Model configurations
DUMMY_MODEL := distilbert/distilgpt2
BASELINE_BACKEND := baseline
HF_BACKEND := hf

# Help target
help: ## Show this help message
	@echo "Coq Proof LLM Benchmark Pipeline"
	@echo "================================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment variables you can override:"
	@echo "  LEMMAS_FILE    Path to lemmas file (default: data/lemmas_auto.jsonl)"
	@echo "  OUTPUT_DIR     Output directory (default: results/batch)"
	@echo "  K_SAMPLES      Number of proof samples per lemma (default: 5)"
	@echo "  MAX_TOKENS     Maximum tokens to generate (default: 256)"
	@echo "  TEMPERATURE    Sampling temperature (default: 0.7)"
	@echo "  TOP_P          Top-p sampling parameter (default: 0.9)"
	@echo "  TIMEOUT        Coq verification timeout in seconds (default: 15)"
	@echo "  LIMIT          Limit number of lemmas to process (default: 0 = all)"
	@echo ""
	@echo "Examples:"
	@echo "  make dataset                    # Generate dataset with defaults"
	@echo "  make proofs-baseline            # Generate baseline proofs"
	@echo "  make proofs-dummy               # Generate proofs with distilgpt2"
	@echo "  make proofs MODEL=llama-3.1     # Generate proofs with custom model"
	@echo "  make verify                     # Verify all proofs in results/batch"
	@echo "  make stats                      # Calculate statistics for all results"
	@echo "  make visualize                  # Create comprehensive visualization plots"
	@echo "  make stats-visualize            # Calculate stats and create visualizations"
	@echo "  make pipeline-baseline          # Run full pipeline with baseline"
	@echo "  make pipeline-dummy             # Run full pipeline with distilgpt2"
	@echo "  make pipeline MODEL=llama-3.1   # Run full pipeline with custom model"

# Dataset generation
dataset: ## Generate synthetic dataset
	@echo "Generating synthetic dataset..."
	$(PYTHON) $(MODULE_PREFIX)generate_dataset --out $(LEMMAS_FILE)
	@echo "Dataset generated: $(LEMMAS_FILE)"

# Proof generation targets
proofs-baseline: ## Generate proofs using baseline backend
	@echo "Generating baseline proofs..."
	$(VENV_ACTIVATE) $(PYTHON) $(MODULE_PREFIX)generate_proofs \
		--lemmas $(LEMMAS_FILE) \
		--backend $(BASELINE_BACKEND) \
		--model "" \
		--outdir $(OUTPUT_DIR) \
		-k $(K_SAMPLES) \
		--max_new_tokens $(MAX_TOKENS) \
		--temperature $(TEMPERATURE) \
		--top_p $(TOP_P) \
		--limit $(LIMIT)

proofs-dummy: ## Generate proofs using distilgpt2 (dummy model)
	@echo "Generating proofs with distilgpt2..."
	$(VENV_ACTIVATE) $(PYTHON) $(MODULE_PREFIX)generate_proofs \
		--lemmas $(LEMMAS_FILE) \
		--backend $(HF_BACKEND) \
		--model $(DUMMY_MODEL) \
		--outdir $(OUTPUT_DIR) \
		-k $(K_SAMPLES) \
		--max_new_tokens $(MAX_TOKENS) \
		--temperature $(TEMPERATURE) \
		--top_p $(TOP_P) \
		--limit $(LIMIT)

proofs: ## Generate proofs with custom model (set MODEL=model_id)
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL variable not set. Use: make proofs MODEL=model_id"; \
		echo "Example: make proofs MODEL=meta-llama/Llama-3.1-8B-Instruct"; \
		exit 1; \
	fi
	@echo "Generating proofs with model: $(MODEL)"
	$(VENV_ACTIVATE) $(PYTHON) $(MODULE_PREFIX)generate_proofs \
		--lemmas $(LEMMAS_FILE) \
		--backend $(HF_BACKEND) \
		--model $(MODEL) \
		--outdir $(OUTPUT_DIR) \
		-k $(K_SAMPLES) \
		--max_new_tokens $(MAX_TOKENS) \
		--temperature $(TEMPERATURE) \
		--top_p $(TOP_P) \
		--limit $(LIMIT)

# Proof verification
verify: ## Verify all proofs in the output directory
	@echo "Verifying proofs in $(OUTPUT_DIR)..."
	$(VENV_ACTIVATE) $(PYTHON) $(MODULE_PREFIX)verify_proofs \
		--proofs_dir $(OUTPUT_DIR) \
		--timeout $(TIMEOUT)

verify-file: ## Verify a specific proofs file (set PROOFS_FILE=path)
	@if [ -z "$(PROOFS_FILE)" ]; then \
		echo "Error: PROOFS_FILE variable not set. Use: make verify-file PROOFS_FILE=path"; \
		echo "Example: make verify-file PROOFS_FILE=results/batch/baseline/proofs.jsonl"; \
		exit 1; \
	fi
	@echo "Verifying proofs file: $(PROOFS_FILE)"
	$(VENV_ACTIVATE) $(PYTHON) $(MODULE_PREFIX)verify_proofs \
		--proofs $(PROOFS_FILE) \
		--timeout $(TIMEOUT)

# Statistics calculation
stats: ## Calculate statistics for all results
	@echo "Calculating statistics for $(OUTPUT_DIR)..."
	$(VENV_ACTIVATE) $(PYTHON) $(MODULE_PREFIX)calculate_stats \
		--results $(OUTPUT_DIR)

stats-file: ## Calculate statistics for a specific results file (set RESULTS_FILE=path)
	@if [ -z "$(RESULTS_FILE)" ]; then \
		echo "Error: RESULTS_FILE variable not set. Use: make stats-file RESULTS_FILE=path"; \
		echo "Example: make stats-file RESULTS_FILE=results/batch/baseline/results.jsonl"; \
		exit 1; \
	fi
	@echo "Calculating statistics for: $(RESULTS_FILE)"
	$(VENV_ACTIVATE) $(PYTHON) $(MODULE_PREFIX)calculate_stats \
		--results $(RESULTS_FILE)

# Visualization targets
visualize: ## Create comprehensive visualization plots
	@echo "Creating visualization plots..."
	$(VENV_ACTIVATE) $(PYTHON) $(MODULE_PREFIX)visualize_results
	@echo "Visualization plots created in plots/ directory!"

stats-visualize: stats visualize ## Calculate stats and create visualizations
	@echo "Statistics calculated and visualizations created!"

# Full pipeline targets
pipeline-baseline: dataset proofs-baseline verify stats ## Run complete pipeline with baseline backend
	@echo "Baseline pipeline completed successfully!"

pipeline-dummy: dataset proofs-dummy verify stats ## Run complete pipeline with distilgpt2
	@echo "DistilGPT2 pipeline completed successfully!"

pipeline: dataset proofs verify stats ## Run complete pipeline with custom model (set MODEL=model_id)
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL variable not set. Use: make pipeline MODEL=model_id"; \
		echo "Example: make pipeline MODEL=meta-llama/Llama-3.1-8B-Instruct"; \
		exit 1; \
	fi
	@echo "Pipeline with $(MODEL) completed successfully!"

# Utility targets
clean: ## Clean generated files and results
	@echo "Cleaning generated files..."
	rm -rf $(OUTPUT_DIR)
	@echo "Cleanup completed!"

clean-dataset: ## Clean only the generated dataset
	@echo "Cleaning dataset..."
	rm -f $(LEMMAS_FILE)
	@echo "Dataset cleaned!"

list-models: ## List available models from models files
	@echo "Available models:"
	@echo "Baseline: (built-in proofs)"
	@echo "Dummy: $(DUMMY_MODEL)"
	@if [ -f models.txt ]; then \
		echo "Models from models.txt:"; \
		cat models.txt; \
	fi
	@if [ -f models_test.txt ]; then \
		echo "Models from models_test.txt:"; \
		cat models_test.txt; \
	fi
	@if [ -f models_full.txt ]; then \
		echo "Models from models_full.txt:"; \
		cat models_full.txt; \
	fi

# Quick test targets
test-baseline: ## Quick test with baseline (1 lemma, 1 sample)
	@echo "Running quick baseline test..."
	$(MAKE) proofs-baseline K_SAMPLES=1 LIMIT=1
	$(MAKE) verify-file PROOFS_FILE=$(OUTPUT_DIR)/baseline/proofs.jsonl
	$(MAKE) stats-file RESULTS_FILE=$(OUTPUT_DIR)/baseline/results.jsonl

test-dummy: ## Quick test with distilgpt2 (1 lemma, 1 sample)
	@echo "Running quick distilgpt2 test..."
	$(MAKE) proofs-dummy K_SAMPLES=1 LIMIT=1
	$(MAKE) verify-file PROOFS_FILE=$(OUTPUT_DIR)/distilbert_distilgpt2/proofs.jsonl
	$(MAKE) stats-file RESULTS_FILE=$(OUTPUT_DIR)/distilbert_distilgpt2/results.jsonl

# Development targets
check: ## Check if all required files exist
	@echo "Checking required files..."
	@test -f requirements.txt || (echo "Error: requirements.txt not found" && exit 1)
	@test -d src || (echo "Error: src directory not found" && exit 1)
	@test -f src/__init__.py || (echo "Error: src/__init__.py not found" && exit 1)
	@echo "All required files found!"

install: ## Install dependencies (requires virtual environment)
	@echo "Installing dependencies..."
	$(VENV_ACTIVATE) pip install -r requirements.txt
	@echo "Dependencies installed!"

# Information targets
info: ## Show current configuration
	@echo "Current configuration:"
	@echo "  Python: $(PYTHON)"
	@echo "  Lemmas file: $(LEMMAS_FILE)"
	@echo "  Output directory: $(OUTPUT_DIR)"
	@echo "  K samples: $(K_SAMPLES)"
	@echo "  Max tokens: $(MAX_TOKENS)"
	@echo "  Temperature: $(TEMPERATURE)"
	@echo "  Top-p: $(TOP_P)"
	@echo "  Timeout: $(TIMEOUT)"
	@echo "  Limit: $(LIMIT)"
	@echo "  Dummy model: $(DUMMY_MODEL)"

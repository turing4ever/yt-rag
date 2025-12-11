# yt-rag Makefile
# Run `make help` for usage information

.PHONY: help install build lint format test test-gen test-report clean status

# Default target
help:
	@echo "yt-rag Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install dependencies (requires uv)"
	@echo "  make ollama       Pull required Ollama models"
	@echo ""
	@echo "Pipeline:"
	@echo "  make build        Run full update pipeline (sync, meta, transcripts, process, embed)"
	@echo "  make build-test   Run pipeline in test mode (5 videos per channel)"
	@echo "  make build-openai Run pipeline using OpenAI API"
	@echo ""
	@echo "Development:"
	@echo "  make lint         Run linter and formatter (auto-fix)"
	@echo "  make format       Alias for make lint"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run benchmark tests with local Ollama"
	@echo "  make test-openai  Run benchmark tests with OpenAI"
	@echo "  make test-gen     Generate benchmark test cases (3-step workflow)"
	@echo "  make test-report  Generate HTML report from test results"
	@echo ""
	@echo "Utilities:"
	@echo "  make status       Show database statistics"
	@echo "  make clean        Remove generated files and caches"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Python 3.14"
	@echo "  - uv (https://docs.astral.sh/uv/)"
	@echo "  - Ollama running locally (for default mode)"
	@echo "  - ~/.yt-rag/.env with OPENAI_API_KEY (for --openai mode)"

# =============================================================================
# Setup
# =============================================================================

install:
	@echo "Installing dependencies..."
	uv sync
	@echo ""
	@echo "Done! Next steps:"
	@echo "  1. Run 'make ollama' to pull required models"
	@echo "  2. Add a channel: yt-rag add https://www.youtube.com/@SomeChannel"
	@echo "  3. Run 'make build' to process videos"

ollama:
	@echo "Pulling required Ollama models..."
	ollama pull mxbai-embed-large
	ollama pull qwen2.5:7b-instruct
	@echo ""
	@echo "Models ready! You can now run 'make build'"

# =============================================================================
# Pipeline
# =============================================================================

# Full pipeline with local Ollama (default)
build:
	@echo "Running full update pipeline..."
	yt-rag update
	@echo ""
	@echo "Pipeline complete! Run 'make status' to see statistics."

# Test mode: only process 5 videos per channel
build-test:
	@echo "Running pipeline in test mode (5 videos per channel)..."
	yt-rag update --test
	@echo ""
	@echo "Test pipeline complete!"

# Full pipeline with OpenAI
build-openai:
	@echo "Running full update pipeline with OpenAI..."
	@if [ ! -f ~/.yt-rag/.env ] || ! grep -q "OPENAI_API_KEY" ~/.yt-rag/.env 2>/dev/null; then \
		echo "Error: OPENAI_API_KEY not found in ~/.yt-rag/.env"; \
		echo "Create the file with: OPENAI_API_KEY=sk-..."; \
		exit 1; \
	fi
	yt-rag update --openai
	@echo ""
	@echo "Pipeline complete!"

# =============================================================================
# Development
# =============================================================================

# Lint and auto-fix issues
lint:
	@echo "Running ruff formatter..."
	uv run ruff format src/
	@echo "Running ruff linter with auto-fix..."
	uv run ruff check --fix src/
	@echo ""
	@echo "Lint complete!"

# Alias for lint
format: lint

# =============================================================================
# Testing
# =============================================================================

# Default test data file
TEST_DATA ?= tests/data/benchmark_generated_gpt-4o.json
TEST_OUTPUT ?= tests/data/benchmark_results.json

# Run benchmark tests with local Ollama
test:
	@echo "Running benchmark tests with local Ollama..."
	@if [ ! -f "$(TEST_DATA)" ]; then \
		echo "Error: Test data not found at $(TEST_DATA)"; \
		echo "Run 'make test-gen' first to generate test cases"; \
		exit 1; \
	fi
	yt-rag test -d "$(TEST_DATA)" -o "$(TEST_OUTPUT)" -v
	@echo ""
	@echo "Results saved to $(TEST_OUTPUT)"
	@echo "Run 'make test-report' to generate HTML report"

# Run benchmark tests with OpenAI
test-openai:
	@echo "Running benchmark tests with OpenAI..."
	@if [ ! -f "$(TEST_DATA)" ]; then \
		echo "Error: Test data not found at $(TEST_DATA)"; \
		echo "Run 'make test-gen' first to generate test cases"; \
		exit 1; \
	fi
	yt-rag test -d "$(TEST_DATA)" -o "$(TEST_OUTPUT)" --openai -v
	@echo ""
	@echo "Results saved to $(TEST_OUTPUT)"

# Run tests with both local and OpenAI validation (comparison mode)
test-compare:
	@echo "Running benchmark tests with validator comparison..."
	yt-rag test -d "$(TEST_DATA)" -o "$(TEST_OUTPUT)" --validate-openai -v
	@echo ""
	@echo "Results saved to $(TEST_OUTPUT)"

# Generate benchmark test cases
# This is a 3-step workflow:
#   1. prepare: Sample videos from your library
#   2. analyze: LLM extracts entities, topics, comparisons
#   3. build: Generate test queries from analysis
#
# By default uses local Ollama. For better quality, use OpenAI:
#   make test-gen-openai
#
# You can also run steps individually:
#   make test-gen-prepare    # Step 1 only
#   make test-gen-analyze    # Step 2 only
#   make test-gen-build      # Step 3 only
test-gen:
	@echo "Generating benchmark test cases (3-step workflow)..."
	@echo ""
	@echo "Step 1/3: Sampling videos from library..."
	yt-rag test-generate --step=prepare
	@echo ""
	@echo "Step 2/3: Analyzing videos with LLM..."
	yt-rag test-generate --step=analyze
	@echo ""
	@echo "Step 3/3: Building test cases..."
	yt-rag test-generate --step=build
	@echo ""
	@echo "Test generation complete!"
	@echo "Run 'make test' to execute the benchmark"

# Generate test cases with OpenAI (higher quality)
test-gen-openai:
	@echo "Generating benchmark test cases with OpenAI..."
	yt-rag test-generate --openai
	@echo ""
	@echo "Test generation complete!"

# Individual test-gen steps (for debugging/customization)
test-gen-prepare:
	@echo "Step 1: Sampling videos..."
	yt-rag test-generate --step=prepare

test-gen-analyze:
	@echo "Step 2: Analyzing videos..."
	yt-rag test-generate --step=analyze

test-gen-analyze-openai:
	@echo "Step 2: Analyzing videos with OpenAI..."
	yt-rag test-generate --step=analyze --openai

test-gen-build:
	@echo "Step 3: Building test cases..."
	yt-rag test-generate --step=build

# Generate HTML report from test results
test-report:
	@echo "Generating HTML report..."
	yt-rag test-report
	@echo ""
	@echo "Report generated! Open tests/data/benchmark_report.html"

# Generate report showing only failures
test-report-failures:
	yt-rag test-report --filter=fail

# Generate report showing validator disagreements
test-report-disagree:
	yt-rag test-report --filter=disagree

# =============================================================================
# Utilities
# =============================================================================

status:
	@echo "Database statistics:"
	@echo ""
	yt-rag status

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf src/**/__pycache__
	rm -rf .venv
	rm -f tests/data/benchmark_results.json
	rm -f tests/data/benchmark_report.html
	@echo "Clean complete!"
	@echo ""
	@echo "Note: Database and indexes are preserved in ~/.yt-rag/"
	@echo "To reset completely, run: rm -rf ~/.yt-rag"

# =============================================================================
# CI/CD helpers
# =============================================================================

# Quick smoke test (for CI)
smoke-test:
	@echo "Running smoke test..."
	yt-rag --help > /dev/null
	yt-rag version
	@echo "Smoke test passed!"

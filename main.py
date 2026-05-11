"""
Main Entry Point — Multi-Agent Research System: Ethical AI in Education

Usage:
  python main.py --mode demo        # Run one example query end-to-end (default)
  python main.py --mode streamlit   # Launch the Streamlit web interface
  python main.py --mode evaluate    # Run the full evaluation pipeline
  python main.py --mode cli         # Run the interactive CLI interface
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO"):
    """Configure logging to both console and file."""
    Path("logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/system.log", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def run_streamlit():
    """Launch the Streamlit web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    print("Open http://localhost:8501 in your browser\n")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"], check=False)


def run_demo():
    """Run one end-to-end demo query and print results."""
    from dotenv import load_dotenv
    import yaml

    # override=True forces .env values to win over any system env vars
    load_dotenv(override=True)
    setup_logging()

    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Import here so dotenv is already loaded
    from src.guardrails.safety_manager import SafetyManager
    from src.autogen_orchestrator import AutoGenOrchestrator

    print("=" * 70)
    print("  MULTI-AGENT RESEARCH SYSTEM — DEMO")
    print("  Topic: Ethical AI in Education")
    print("=" * 70)

    safety_manager = SafetyManager(config.get("safety", {}))

    demo_query = (
        "What are the key ethical considerations when using AI systems in "
        "K-12 education, particularly regarding student data privacy and "
        "algorithmic bias?"
    )
    print(f"\nQuery:\n  {demo_query}\n")

    # --- Input safety check ---
    print("--- Input Safety Check ---")
    input_result = safety_manager.check_input_safety(demo_query)
    if input_result["safe"]:
        print("  [OK] Input passed all safety checks")
    else:
        print("  [WARN] Safety violations detected:")
        for v in input_result["violations"]:
            print(f"    - [{v['severity'].upper()}] {v['reason']}")
        if input_result["action"] == "block":
            print("\n  Query blocked by safety system. Exiting demo.")
            return

    # --- Agent pipeline ---
    print("\n--- Initializing Agent Pipeline ---")
    try:
        orchestrator = AutoGenOrchestrator(config)
        print("  [OK] Research team ready (Planner -> Researcher -> Writer -> Critic)")
    except Exception as exc:
        print(f"  [ERR] Failed to initialize orchestrator: {exc}")
        return

    print("\n--- Running Multi-Agent Research ---")
    print("  (Agents are collaborating - this may take 30-90 seconds...)\n")

    try:
        result = orchestrator.process_query(input_result["query"])
    except Exception as exc:
        print(f"  ✗ Error during query processing: {exc}")
        import traceback
        traceback.print_exc()
        return

    if "error" in result:
        print(f"  [ERR] Orchestrator error: {result['error']}")
        # Still show partial results if any
        response = result.get("response", "No response generated")
    else:
        response = result.get("response", "No response generated")

    # --- Output safety check ---
    print("--- Output Safety Check ---")
    sources = result.get("metadata", {}).get("sources", [])
    output_result = safety_manager.check_output_safety(response, sources)
    if output_result["safe"]:
        print("  [OK] Output passed all safety checks")
    else:
        print(f"  [WARN] Action taken: {output_result['action_taken']}")
        for v in output_result["violations"]:
            print(f"    - [{v['severity'].upper()}] {v['reason']}")

    # --- Print results ---
    print("\n" + "=" * 70)
    print("  RESEARCH RESPONSE")
    print("=" * 70)
    safe_response = output_result.get("response", response)
    print(safe_response[:3000])
    if len(safe_response) > 3000:
        print("\n  [... response truncated for display ...]")

    # --- Metadata ---
    meta = result.get("metadata", {})
    print("\n--- Metadata ---")
    print(f"  Messages exchanged : {meta.get('num_messages', 0)}")
    print(f"  Sources gathered   : {meta.get('num_sources', 0)}")
    agents = meta.get("agents_involved", [])
    if agents:
        print(f"  Agents involved    : {', '.join(agents)}")

    # --- Safety summary ---
    stats = safety_manager.get_safety_stats()
    print("\n--- Safety Summary ---")
    print(f"  Input checks   : {stats['input_checks']}")
    print(f"  Output checks  : {stats['output_checks']}")
    print(f"  Violations     : {stats['violations']}")

    print("\n  [DONE] Demo complete!")


async def run_evaluation():
    """Run the full batch evaluation pipeline."""
    from dotenv import load_dotenv
    import yaml

    load_dotenv(override=True)
    setup_logging()

    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.evaluator import SystemEvaluator

    print("=" * 70)
    print("  EVALUATION PIPELINE — Ethical AI in Education")
    print("=" * 70)

    print("\nInitializing orchestrator...")
    try:
        orchestrator = AutoGenOrchestrator(config)
        print("  [OK] Orchestrator ready")
    except Exception as exc:
        print(f"  [ERR] Failed to initialize orchestrator: {exc}")
        return

    evaluator = SystemEvaluator(config, orchestrator=orchestrator)

    print("\nRunning evaluation on data/example_queries.json ...")
    print("  (Each query runs through the full agent pipeline — this takes several minutes)\n")

    report = await evaluator.evaluate_system("data/example_queries.json")

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    summary = report.get("summary", {})
    print(f"\n  Total queries  : {summary.get('total_queries', 0)}")
    print(f"  Successful     : {summary.get('successful', 0)}")
    print(f"  Failed         : {summary.get('failed', 0)}")
    print(f"  Success rate   : {summary.get('success_rate', 0.0):.1%}")

    scores = report.get("scores", {})
    print(f"\n  Overall average score : {scores.get('overall_average', 0.0):.3f}")

    print("\n  Scores by criterion:")
    for criterion, score in scores.get("by_criterion", {}).items():
        bar = "#" * int(score * 20)
        print(f"    {criterion:<25} {score:.3f}  {bar}")

    best = report.get("best_result")
    worst = report.get("worst_result")
    if best:
        print(f"\n  Best  query (score={best['score']:.3f}): {best['query'][:70]}")
    if worst:
        print(f"  Worst query (score={worst['score']:.3f}): {worst['query'][:70]}")

    print("\n  Results saved to outputs/")


def run_cli():
    """Run the interactive CLI interface."""
    from src.ui.cli import main as cli_main
    cli_main()


# ---------------------------------------------------------------------------
# Argument parsing & dispatch
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant — Ethical AI in Education"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "streamlit", "evaluate", "cli", "web"],
        default="demo",
        help="Execution mode (default: demo)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode in ("streamlit", "web"):
        run_streamlit()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation())
    elif args.mode == "cli":
        run_cli()


if __name__ == "__main__":
    main()

"""CLI commands for the LLM evaluation pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="llm-judge",
    help="LLM evaluation pipeline: inference → autocheck → judge → compare",
)
console = Console()

DEFAULT_CONFIG = "configs/run-config.yaml"


@app.command()
def infer(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run config YAML path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSONL path"),
) -> None:
    """Stage 1: Run candidate model inference."""
    from llm_judge.config import load_run_config
    from llm_judge.stages.inference import run_inference
    from llm_judge.utils import build_run_dir, derive_dataset_layer

    console.print("[bold blue]Stage 1: Inference[/bold blue]")
    cfg = load_run_config(config)
    started_at = datetime.now().astimezone()
    run_dir = None
    if not output:
        layer = derive_dataset_layer(cfg.dataset.testcases_path)
        run_dir = build_run_dir(layer, started_at=started_at)
        console.print(f"[dim]Output dir:[/dim] {run_dir}")
    out = run_inference(config, output, started_at=started_at, run_dir=run_dir)
    console.print(f"[green]Done:[/green] {out}")


@app.command()
def judge(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run config YAML path"),
    inference: Optional[str] = typer.Option(None, "--inference", "-i", help="Inference JSONL path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSONL path"),
) -> None:
    """Stage 3: LLM-as-a-Judge evaluation.

    inference_repeats == 1 → metrics mode (quality scoring)
    inference_repeats >= 2 → consistency mode (consistency scoring)
    """
    from llm_judge.config import load_run_config
    from llm_judge.utils import build_run_dir

    cfg = load_run_config(config)
    started_at = datetime.now().astimezone()
    run_dir = None
    if not output:
        from llm_judge.utils import derive_dataset_layer

        layer = derive_dataset_layer(cfg.dataset.testcases_path)
        run_dir = build_run_dir(layer, started_at=started_at)
        console.print(f"[dim]Output dir:[/dim] {run_dir}")

    if cfg.protocol.repeats.inference_repeats >= 2:
        from llm_judge.stages.consistency import run_consistency

        console.print("[bold blue]Stage 3: Judge (consistency)[/bold blue]")
        out = run_consistency(config, inference, output, started_at=started_at, run_dir=run_dir)
    else:
        from llm_judge.stages.judge import run_judge

        console.print("[bold blue]Stage 3: Judge (metrics)[/bold blue]")
        out = run_judge(config, inference, output, started_at=started_at, run_dir=run_dir)

    console.print(f"[green]Done:[/green] {out}")


@app.command()
def compare(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run config YAML path"),
    judgements: Optional[str] = typer.Option(None, "--judgements", "-j", help="Judgements JSONL path"),
    consistency: Optional[str] = typer.Option(None, "--consistency", help="Consistency JSONL path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON path"),
) -> None:
    """Stage 4: Aggregate and produce comparison report."""
    from llm_judge.config import load_run_config
    from llm_judge.stages.compare import run_compare
    from llm_judge.utils import build_run_dir, derive_dataset_layer

    console.print("[bold blue]Stage 4: Compare[/bold blue]")
    cfg = load_run_config(config)
    started_at = datetime.now().astimezone()
    run_dir = None
    if not output:
        layer = derive_dataset_layer(cfg.dataset.testcases_path)
        run_dir = build_run_dir(layer, started_at=started_at)
        console.print(f"[dim]Output dir:[/dim] {run_dir}")
    out = run_compare(
        config,
        judgements,
        output_path=output,
        consistency_path=consistency,
        started_at=started_at,
        run_dir=run_dir,
    )
    console.print(f"[green]Done:[/green] {out}")
    console.print(f"[green]Markdown:[/green] {out.with_suffix('.md')}")


@app.command()
def run_all(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run config YAML path"),
) -> None:
    """Run all stages: inference → judge → compare."""
    from llm_judge.config import load_run_config
    from llm_judge.stages.autocheck import run_autocheck
    from llm_judge.stages.compare import run_compare
    from llm_judge.stages.inference import run_inference
    from llm_judge.utils import build_run_dir, derive_dataset_layer

    cfg = load_run_config(config)
    consistency_mode = cfg.protocol.repeats.inference_repeats >= 2

    started_at = datetime.now().astimezone()
    layer = derive_dataset_layer(cfg.dataset.testcases_path)
    run_dir = build_run_dir(layer, started_at=started_at)

    console.print(f"[bold magenta]Running full pipeline → {run_dir}[/bold magenta]")

    console.print("\n[bold blue]Stage 1: Inference[/bold blue]")
    inf_out = run_inference(config, started_at=started_at, run_dir=run_dir)
    console.print(f"[green]Done:[/green] {inf_out}")

    # Autocheck (internal: format/schema validation for compare report)
    ac_out = run_autocheck(config, str(inf_out), started_at=started_at, run_dir=run_dir)

    jdg_out = None
    con_out = None

    if consistency_mode:
        from llm_judge.stages.consistency import run_consistency

        console.print("\n[bold blue]Stage 2: Judge (consistency)[/bold blue]")
        con_out = run_consistency(config, str(inf_out), started_at=started_at, run_dir=run_dir)
        console.print(f"[green]Done:[/green] {con_out}")
    else:
        from llm_judge.stages.judge import run_judge

        console.print("\n[bold blue]Stage 2: Judge (metrics)[/bold blue]")
        jdg_out = run_judge(config, str(inf_out), started_at=started_at, run_dir=run_dir)
        console.print(f"[green]Done:[/green] {jdg_out}")

    console.print("\n[bold blue]Stage 3: Compare[/bold blue]")
    cmp_out = run_compare(
        config,
        str(jdg_out) if jdg_out else None,
        consistency_path=str(con_out) if con_out else None,
        started_at=started_at,
        run_dir=run_dir,
    )
    console.print(f"[green]Done:[/green] {cmp_out}")
    console.print(f"[green]Markdown:[/green] {cmp_out.with_suffix('.md')}")

    console.print("\n[bold green]Pipeline complete![/bold green]")

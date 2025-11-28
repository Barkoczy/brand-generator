"""CLI rozhraní pro generátor názvů značek."""

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.agent.orchestrator import GenerationStats, NameGenAgent
from src.config import load_settings
from src.db.repository import CandidateRepository
from src.llm import check_provider_availability, get_default_model, list_providers

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

console = Console()


def progress_callback(message: str, stats: GenerationStats | None = None) -> None:
    """Callback pro zobrazení průběhu."""
    if stats:
        console.print(f"[cyan]{message}[/cyan]")
    else:
        console.print(f"[dim]{message}[/dim]")


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Cesta ke konfiguračnímu souboru (výchozí: config.yaml)",
)
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """Generátor neologismů pro názvy firem a produktů.

    Tento nástroj generuje vymyšlená slova vhodná jako název firmy
    nebo produktu, s důrazem na registrovatelnost ochranných známek.

    Podporuje více LLM providerů: anthropic, openai, ollama, lmstudio, gemini.
    """
    ctx.ensure_object(dict)
    ctx.obj["settings"] = load_settings(config)


@cli.command("generuj")
@click.option("--pocet", "-n", default=100, help="Počet kandidátů k vygenerování")
@click.option("--seed", "-s", type=int, default=None, help="Seed pro reprodukovatelné výsledky")
@click.option("--vystup", "-o", type=click.Path(), default=None, help="Soubor pro export")
@click.pass_context
def generate(ctx: click.Context, pocet: int, seed: int | None, vystup: str | None) -> None:
    """Vygeneruj kandidáty na název (offline režim, bez LLM).

    Generuje názvy podle C/V vzorů a hodnotí je heuristicky.
    Pro LLM hodnocení použij příkaz 'spust'.
    """
    settings = ctx.obj["settings"]

    console.print(f"[bold]Generuji {pocet} kandidátů...[/bold]")

    if seed:
        console.print(f"[dim]Seed: {seed}[/dim]")

    agent = NameGenAgent(settings, use_llm=False, progress_callback=progress_callback)
    candidates = agent.run_offline(pocet)

    if not candidates:
        console.print("[red]Nepodařilo se vygenerovat žádné kandidáty.[/red]")
        return

    # Display results
    table = Table(title=f"Top {min(20, len(candidates))} kandidátů")
    table.add_column("Název", style="bold cyan")
    table.add_column("Vzor")
    table.add_column("Skóre", justify="right")

    for c in candidates[:20]:
        score_color = "green" if c.heuristic_score and c.heuristic_score >= 7 else "yellow"
        score_str = f"{c.heuristic_score:.1f}" if c.heuristic_score else "-"
        table.add_row(c.name, c.pattern, f"[{score_color}]{score_str}[/{score_color}]")

    console.print(table)
    console.print(f"\n[dim]Celkem uloženo: {len(candidates)} kandidátů[/dim]")

    if vystup:
        with open(vystup, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(f"{c.name}\n")
        console.print(f"[green]Export uložen do: {vystup}[/green]")


@cli.command("spust")
@click.option("--cil", "-t", default=10, help="Cílový počet vynikajících kandidátů (skóre >= 8)")
@click.option("--iterace", "-i", default=10, help="Maximální počet iterací")
@click.option("--davka", "-b", default=50, help="Počet kandidátů na iteraci")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "openai", "ollama", "lmstudio", "gemini"]),
    default=None,
    help="LLM provider (přepíše config)",
)
@click.option("--model", "-m", default=None, help="Model name (přepíše config)")
@click.pass_context
def run_agent(
    ctx: click.Context,
    cil: int,
    iterace: int,
    davka: int,
    provider: str | None,
    model: str | None,
) -> None:
    """Spusť autonomního agenta s LLM hodnocením.

    Agent generuje kandidáty, filtruje je heuristicky a nejlepší
    nechává hodnotit LLM. Běží dokud nenajde dostatek
    vynikajících kandidátů nebo nedosáhne limitu iterací.

    Podporované providery:
      - anthropic (Claude) - vyžaduje ANTHROPIC_API_KEY
      - openai (GPT) - vyžaduje OPENAI_API_KEY
      - ollama - lokální, vyžaduje běžící Ollama server
      - lmstudio - lokální, vyžaduje běžící LM Studio
      - gemini - vyžaduje GOOGLE_API_KEY
    """
    settings = ctx.obj["settings"]

    # Override provider/model from CLI if specified
    if provider:
        settings.llm.provider = provider
        if not model:
            settings.llm.model = get_default_model(provider)
    if model:
        settings.llm.model = model

    console.print("[bold]Spouštím autonomního agenta...[/bold]")
    console.print(f"[dim]Provider: {settings.llm.provider}, Model: {settings.llm.model}[/dim]")
    console.print(f"[dim]Cíl: {cil} vynikajících kandidátů[/dim]")
    console.print(f"[dim]Max iterací: {iterace}, dávka: {davka}[/dim]")

    # Check provider availability
    from src.llm.base import LLMConfig

    llm_config = LLMConfig(
        provider=settings.llm.provider,
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        api_base=settings.llm.api_base,
    )
    available, msg = check_provider_availability(llm_config)
    if not available:
        console.print(f"[red]{msg}[/red]")
        return

    try:
        agent = NameGenAgent(settings, use_llm=True, progress_callback=progress_callback)
        results = agent.run(target_excellent=cil, max_iterations=iterace, batch_size=davka)
    except Exception as e:
        console.print(f"[red]Chyba: {e}[/red]")
        return

    if not results:
        console.print("[yellow]Nebyly nalezeny žádné vynikající kandidáty.[/yellow]")
        return

    # Display results
    table = Table(title="Vynikající kandidáti (skóre >= 8)")
    table.add_column("Název", style="bold cyan")
    table.add_column("Skóre", justify="right")
    table.add_column("Kategorie")
    table.add_column("Doporučení")

    for c in results:
        score_str = str(c.llm_score) if c.llm_score else "-"
        table.add_row(
            c.name,
            f"[green]{score_str}[/green]",
            c.category or "-",
            (c.recommendation[:50] + "...") if c.recommendation and len(c.recommendation) > 50 else (c.recommendation or "-"),
        )

    console.print(table)


@cli.command("providery")
def list_available_providers() -> None:
    """Zobraz dostupné LLM providery a jejich stav."""
    from src.llm.base import LLMConfig

    console.print("\n[bold]Dostupní LLM provideři:[/bold]\n")

    providers_info = [
        ("anthropic", "Claude modely", "ANTHROPIC_API_KEY"),
        ("openai", "GPT modely", "OPENAI_API_KEY"),
        ("ollama", "Lokální modely", "http://localhost:11434"),
        ("lmstudio", "Lokální modely", "http://localhost:1234/v1"),
        ("gemini", "Google Gemini", "GOOGLE_API_KEY"),
    ]

    table = Table()
    table.add_column("Provider", style="cyan")
    table.add_column("Popis")
    table.add_column("Výchozí model")
    table.add_column("Požadavek")
    table.add_column("Stav")

    for name, desc, requirement in providers_info:
        default_model = get_default_model(name)
        config = LLMConfig(provider=name, model=default_model)
        available, _ = check_provider_availability(config)
        status = "[green]OK[/green]" if available else "[red]Nedostupný[/red]"
        table.add_row(name, desc, default_model, requirement, status)

    console.print(table)
    console.print("\n[dim]Tip: Nastav provider v config.yaml nebo použij --provider při spuštění.[/dim]")


@cli.command("top")
@click.option("--limit", "-l", default=20, help="Počet zobrazených kandidátů")
@click.option("--min-skore", "-m", type=int, default=None, help="Minimální skóre")
@click.pass_context
def show_top(ctx: click.Context, limit: int, min_skore: int | None) -> None:
    """Zobraz nejlepší kandidáty z databáze."""
    settings = ctx.obj["settings"]
    repo = CandidateRepository(settings)

    candidates = repo.get_top(limit=limit, min_score=min_skore)

    if not candidates:
        console.print("[yellow]Žádní kandidáti v databázi.[/yellow]")
        return

    table = Table(title=f"Top {len(candidates)} kandidátů")
    table.add_column("Název", style="bold cyan")
    table.add_column("LLM Skóre", justify="right")
    table.add_column("Heur. Skóre", justify="right")
    table.add_column("Kategorie")
    table.add_column("Status")

    for c in candidates:
        llm_str = str(c.llm_score) if c.llm_score else "-"
        heur_str = f"{c.heuristic_score:.1f}" if c.heuristic_score else "-"
        status_color = "green" if c.status == "favorite" else ("red" if c.status == "rejected" else "dim")
        table.add_row(
            c.name,
            f"[green]{llm_str}[/green]",
            heur_str,
            c.category or "-",
            f"[{status_color}]{c.status}[/{status_color}]",
        )

    console.print(table)


@cli.command("detail")
@click.argument("nazev")
@click.pass_context
def show_detail(ctx: click.Context, nazev: str) -> None:
    """Zobraz detail konkrétního kandidáta."""
    settings = ctx.obj["settings"]
    repo = CandidateRepository(settings)

    candidate = repo.get_by_name(nazev)

    if not candidate:
        console.print(f"[red]Kandidát '{nazev}' nenalezen.[/red]")
        return

    console.print(f"\n[bold cyan]{candidate.name}[/bold cyan]")
    console.print(f"[dim]Vzor: {candidate.pattern}[/dim]")
    console.print(f"[dim]Status: {candidate.status}[/dim]")
    console.print()

    if candidate.llm_score is not None:
        console.print(f"[bold]LLM Skóre:[/bold] [green]{candidate.llm_score}/10[/green]")
        console.print(f"[bold]Kategorie:[/bold] {candidate.category}")

    if candidate.heuristic_score is not None:
        console.print(f"[bold]Heuristické skóre:[/bold] {candidate.heuristic_score:.1f}/10")

    if candidate.pros:
        console.print("\n[bold green]Silné stránky:[/bold green]")
        for pro in candidate.pros:
            console.print(f"  + {pro}")

    if candidate.cons:
        console.print("\n[bold red]Slabé stránky:[/bold red]")
        for con in candidate.cons:
            console.print(f"  - {con}")

    if candidate.flags:
        console.print("\n[bold yellow]Varování:[/bold yellow]")
        for flag in candidate.flags:
            console.print(f"  ! {flag}")

    if candidate.recommendation:
        console.print(f"\n[bold]Doporučení:[/bold] {candidate.recommendation}")


@cli.command("oblibene")
@click.argument("nazev")
@click.pass_context
def mark_favorite(ctx: click.Context, nazev: str) -> None:
    """Označ kandidáta jako oblíbeného."""
    settings = ctx.obj["settings"]
    repo = CandidateRepository(settings)

    if not repo.exists(nazev):
        console.print(f"[red]Kandidát '{nazev}' nenalezen.[/red]")
        return

    repo.set_status(nazev, "favorite")
    console.print(f"[green]'{nazev}' označen jako oblíbený.[/green]")


@cli.command("vyrad")
@click.argument("nazev")
@click.pass_context
def mark_rejected(ctx: click.Context, nazev: str) -> None:
    """Vyřaď kandidáta."""
    settings = ctx.obj["settings"]
    repo = CandidateRepository(settings)

    if not repo.exists(nazev):
        console.print(f"[red]Kandidát '{nazev}' nenalezen.[/red]")
        return

    repo.set_status(nazev, "rejected")
    console.print(f"[yellow]'{nazev}' vyřazen.[/yellow]")


@cli.command("statistiky")
@click.pass_context
def show_stats(ctx: click.Context) -> None:
    """Zobraz statistiky databáze."""
    settings = ctx.obj["settings"]
    repo = CandidateRepository(settings)

    total = repo.count()
    favorites = repo.count(status="favorite")
    rejected = repo.count(status="rejected")
    excellent = repo.count_above_score(8)

    console.print("\n[bold]Statistiky databáze[/bold]")
    console.print(f"  Celkem kandidátů: {total}")
    console.print(f"  Vynikajících (skóre >= 8): [green]{excellent}[/green]")
    console.print(f"  Oblíbených: [cyan]{favorites}[/cyan]")
    console.print(f"  Vyřazených: [red]{rejected}[/red]")


@cli.command("export")
@click.argument("soubor")
@click.option("--min-skore", "-m", type=int, default=6, help="Minimální skóre pro export")
@click.option("--format", "-f", type=click.Choice(["txt", "csv", "json"]), default="txt")
@click.pass_context
def export_candidates(ctx: click.Context, soubor: str, min_skore: int, format: str) -> None:
    """Exportuj kandidáty do souboru."""
    import json as json_module

    settings = ctx.obj["settings"]
    repo = CandidateRepository(settings)

    candidates = repo.get_all_scored(limit=1000, min_heuristic=float(min_skore))

    if not candidates:
        console.print("[yellow]Žádní kandidáti k exportu.[/yellow]")
        return

    with open(soubor, "w", encoding="utf-8") as f:
        if format == "txt":
            for c in candidates:
                f.write(f"{c.name}\n")
        elif format == "csv":
            f.write("name,llm_score,heuristic_score,category,status\n")
            for c in candidates:
                f.write(
                    f"{c.name},{c.llm_score or ''},{c.heuristic_score or ''},{c.category or ''},{c.status}\n"
                )
        elif format == "json":
            data = [
                {
                    "name": c.name,
                    "llm_score": c.llm_score,
                    "heuristic_score": c.heuristic_score,
                    "category": c.category,
                    "pros": c.pros,
                    "cons": c.cons,
                    "flags": c.flags,
                    "recommendation": c.recommendation,
                    "status": c.status,
                }
                for c in candidates
            ]
            json_module.dump(data, f, ensure_ascii=False, indent=2)

    console.print(f"[green]Exportováno {len(candidates)} kandidátů do {soubor}[/green]")


def main() -> None:
    """Entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()

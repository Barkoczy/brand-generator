"""CLI rozhraní pro generátor názvů značek."""

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.agent.orchestrator import GenerationStats, NameGenAgent
from src.config import load_settings
from src.db.repository import CandidateRepository
from src.generator import GeneticConfig, GeneticOptimizer, IslandConfig, IslandOptimizer
from src.generator.genetic_optimizer import GenerationStats as GeneticStats
from src.llm import check_provider_availability, get_default_model, list_providers
from src.scoring import CombinedPhoneticScorer
from src.trademark.clearance_checker import create_checker_from_settings

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


# =============================================================================
# ADVANCED COMMANDS - Genetic Algorithm & Phonetic Analysis
# =============================================================================


@cli.command("evoluce")
@click.option("--populace", "-p", default=200, help="Velikost populace")
@click.option("--generace", "-g", default=50, help="Počet generací")
@click.option("--mutace", "-m", default=0.15, type=float, help="Míra mutace (0-1)")
@click.option("--cil", "-t", default=8.0, type=float, help="Cílové skóre")
@click.option("--ostrovy", "-o", default=1, help="Počet ostrovů (paralelní evoluce)")
@click.option("--vystup", type=click.Path(), default=None, help="Soubor pro export")
@click.pass_context
def run_evolution(
    ctx: click.Context,
    populace: int,
    generace: int,
    mutace: float,
    cil: float,
    ostrovy: int,
    vystup: str | None,
) -> None:
    """Spusť genetický algoritmus pro optimalizaci názvů.

    Evoluční přístup využívající:
    - Selekci nejlepších kandidátů (turnajová)
    - Křížení (crossover) úspěšných názvů
    - Mutaci pro diverzitu
    - Elitismus (zachování nejlepších)

    Příklady:
        brand-gen evoluce --populace 500 --generace 100
        brand-gen evoluce --ostrovy 4 --cil 9.0
    """
    settings = ctx.obj["settings"]

    console.print("[bold]🧬 Spouštím genetický algoritmus...[/bold]")
    console.print(f"[dim]Populace: {populace}, Generace: {generace}[/dim]")
    console.print(f"[dim]Mutace: {mutace:.0%}, Cílové skóre: {cil}[/dim]")

    if ostrovy > 1:
        console.print(f"[dim]Ostrovní model: {ostrovy} ostrovů[/dim]")

    genetic_config = GeneticConfig(
        population_size=populace,
        generations=generace,
        mutation_rate=mutace,
        target_score=cil,
    )

    def progress_callback(stats: GeneticStats) -> None:
        """Display generation progress."""
        console.print(
            f"[cyan]Gen {stats.generation:3d}[/cyan] | "
            f"Best: [green]{stats.best_fitness:.2f}[/green] | "
            f"Avg: {stats.avg_fitness:.2f} | "
            f"Best word: [bold]{stats.best_word}[/bold]"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evoluce probíhá...", total=None)

        if ostrovy > 1:
            island_config = IslandConfig(num_islands=ostrovy)
            optimizer = IslandOptimizer(settings, genetic_config, island_config)
        else:
            optimizer = GeneticOptimizer(
                settings, genetic_config, progress_callback=progress_callback
            )

        candidates = optimizer.get_best_candidates(count=20)
        progress.update(task, completed=True)

    if not candidates:
        console.print("[red]Nepodařilo se vygenerovat kandidáty.[/red]")
        return

    # Score candidates with phonetic models
    scorer = CombinedPhoneticScorer()

    table = Table(title="🏆 Top kandidáti z evoluce")
    table.add_column("#", style="dim")
    table.add_column("Název", style="bold cyan")
    table.add_column("Vzor")
    table.add_column("Celkové", justify="right")
    table.add_column("Markov", justify="right")
    table.add_column("Sonorita", justify="right")
    table.add_column("Unikátnost", justify="right")

    for i, c in enumerate(candidates[:20], 1):
        result = scorer.score(c.name)
        score_color = "green" if result.total_score >= 8 else "yellow"
        table.add_row(
            str(i),
            c.name,
            c.pattern,
            f"[{score_color}]{result.total_score:.1f}[/{score_color}]",
            f"{result.markov_score:.1f}",
            f"{result.sonority_score:.1f}",
            f"{result.uniqueness_score:.1f}",
        )

    console.print(table)

    if vystup:
        with open(vystup, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(f"{c.name}\n")
        console.print(f"[green]Export uložen do: {vystup}[/green]")


@cli.command("analyzuj")
@click.argument("nazev")
@click.pass_context
def analyze_phonetics(ctx: click.Context, nazev: str) -> None:
    """Zobraz detailní fonetickou analýzu názvu.

    Analyzuje název pomocí matematických modelů:
    - Markovovy řetězce (přirozenost sekvencí)
    - Sonority Sequencing Principle (vyslovitelnost)
    - Shannon entropie (vyváženost vzorů)
    - Fonotaktická pravděpodobnost
    - Zipfův zákon (frekvence fonémů)
    - Levenshteinova vzdálenost (unikátnost)

    Příklady:
        brand-gen analyzuj Valujo
        brand-gen analyzuj Nextra
    """
    scorer = CombinedPhoneticScorer()
    result = scorer.score(nazev)

    console.print()
    console.print(Panel(f"[bold cyan]{nazev}[/bold cyan]", title="Fonetická analýza"))

    # Overall score
    score_color = "green" if result.total_score >= 8 else ("yellow" if result.total_score >= 6 else "red")
    console.print(f"\n[bold]Celkové skóre:[/bold] [{score_color}]{result.total_score:.2f}/10[/{score_color}]")

    # Detailed scores table
    table = Table(title="Detailní skóre", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Skóre", justify="right")
    table.add_column("Váha", justify="right")
    table.add_column("Příspěvek", justify="right")
    table.add_column("Popis")

    weights = scorer.weights
    models_info = [
        ("Markov (n-gramy)", result.markov_score, weights["markov"], "Přirozenost sekvencí písmen"),
        ("Sonorita (SSP)", result.sonority_score, weights["sonority"], "Soulad se slabičnou strukturou"),
        ("Entropie", result.entropy_score, weights["entropy"], "Vyváženost vzorů (ne příliš náhodné/repetitivní)"),
        ("Fonotaktika", result.phonotactic_score, weights["phonotactic"], "Pravděpodobnost zvukových kombinací"),
        ("Zipfův zákon", result.zipf_score, weights["zipf"], "Použití běžných vs vzácných fonémů"),
        ("Unikátnost", result.uniqueness_score, weights["uniqueness"], "Vzdálenost od existujících slov/značek"),
    ]

    for name, score, weight, desc in models_info:
        contribution = score * weight
        score_color = "green" if score >= 7 else ("yellow" if score >= 5 else "red")
        table.add_row(
            name,
            f"[{score_color}]{score:.1f}[/{score_color}]",
            f"{weight:.0%}",
            f"{contribution:.2f}",
            desc,
        )

    console.print(table)

    # Interpretation
    console.print("\n[bold]Interpretace:[/bold]")

    if result.markov_score >= 7:
        console.print("  [green]✓[/green] Sekvence písmen znějí přirozeně")
    elif result.markov_score < 4:
        console.print("  [red]✗[/red] Sekvence písmen znějí nepřirozeně")

    if result.sonority_score >= 7:
        console.print("  [green]✓[/green] Snadno vyslovitelné")
    elif result.sonority_score < 4:
        console.print("  [red]✗[/red] Obtížná výslovnost")

    if result.uniqueness_score >= 8:
        console.print("  [green]✓[/green] Vysoce unikátní název")
    elif result.uniqueness_score < 5:
        console.print("  [yellow]![/yellow] Podobné existujícím slovům")

    if result.entropy_score >= 7:
        console.print("  [green]✓[/green] Dobře vyvážená struktura")
    elif result.entropy_score < 4:
        console.print("  [yellow]![/yellow] Příliš repetitivní nebo náhodné")

    # Recommendation
    console.print()
    if result.total_score >= 8:
        console.print("[bold green]Doporučení: Výborný kandidát pro značku![/bold green]")
    elif result.total_score >= 6:
        console.print("[bold yellow]Doporučení: Přijatelný kandidát, zvažte alternativy.[/bold yellow]")
    else:
        console.print("[bold red]Doporučení: Slabý kandidát, hledejte lepší alternativy.[/bold red]")


@cli.command("porovnej")
@click.argument("nazvy", nargs=-1, required=True)
@click.pass_context
def compare_names(ctx: click.Context, nazvy: tuple[str, ...]) -> None:
    """Porovnej více názvů vedle sebe.

    Zobrazí srovnávací tabulku s fonetickými skóre pro všechny zadané názvy.

    Příklady:
        brand-gen porovnej Valujo Nextra Zonify
        brand-gen porovnej Google Apple Amazon
    """
    if len(nazvy) < 2:
        console.print("[red]Zadejte alespoň 2 názvy k porovnání.[/red]")
        return

    scorer = CombinedPhoneticScorer()
    results = [(name, scorer.score(name)) for name in nazvy]

    # Sort by total score
    results.sort(key=lambda x: x[1].total_score, reverse=True)

    table = Table(title="Srovnání názvů")
    table.add_column("Pořadí", style="dim")
    table.add_column("Název", style="bold cyan")
    table.add_column("Celkové", justify="right")
    table.add_column("Markov", justify="right")
    table.add_column("Sonorita", justify="right")
    table.add_column("Entropie", justify="right")
    table.add_column("Fonotak.", justify="right")
    table.add_column("Zipf", justify="right")
    table.add_column("Unikátnost", justify="right")

    for i, (name, result) in enumerate(results, 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
        score_color = "green" if result.total_score >= 8 else ("yellow" if result.total_score >= 6 else "red")
        table.add_row(
            medal,
            name,
            f"[{score_color}]{result.total_score:.1f}[/{score_color}]",
            f"{result.markov_score:.1f}",
            f"{result.sonority_score:.1f}",
            f"{result.entropy_score:.1f}",
            f"{result.phonotactic_score:.1f}",
            f"{result.zipf_score:.1f}",
            f"{result.uniqueness_score:.1f}",
        )

    console.print(table)

    # Winner announcement
    winner_name, winner_result = results[0]
    console.print(f"\n[bold green]🏆 Nejlepší kandidát: {winner_name} ({winner_result.total_score:.1f}/10)[/bold green]")


# =============================================================================
# TRADEMARK CLEARANCE COMMANDS
# =============================================================================


@cli.command("clearance")
@click.argument("nazev")
@click.option(
    "--tridy",
    "-t",
    type=str,
    default="",
    help="Seznam tříd Nice oddělených čárkou (např. 9,35,42)",
)
@click.pass_context
def check_trademark(ctx: click.Context, nazev: str, tridy: str) -> None:
    """Zkontroluj kolize názvu v databázi ochranných známek.

    Prohledá databázi EUIPO a vrátí rizikové hodnocení.
    Bez API klíčů použije simulovaná data (mock).

    Příklady:
        brand-gen clearance MYNAME
        brand-gen clearance MYNAME --tridy 9,35,42
    """
    settings = ctx.obj["settings"]

    # Parse Nice classes
    if tridy:
        nice_classes = [int(c.strip()) for c in tridy.split(",") if c.strip()]
    else:
        nice_classes = settings.trademark.default_nice_classes

    console.print(f"\n[bold]Kontroluji název: {nazev}[/bold]")
    console.print(f"[dim]Třídy Nice: {', '.join(map(str, nice_classes))}[/dim]")

    # Create checker - use env vars as fallback if not in config
    import os
    client_id = settings.trademark.euipo.client_id or os.getenv("EUIPO_CLIENT_ID")
    client_secret = settings.trademark.euipo.client_secret or os.getenv("EUIPO_CLIENT_SECRET")

    checker = create_checker_from_settings(
        client_id=client_id,
        client_secret=client_secret,
        sandbox=settings.trademark.euipo.sandbox,
        similarity_threshold=settings.trademark.similarity_threshold,
    )

    if checker.is_using_mock:
        console.print("[yellow]⚠ Používám simulovaná data (EUIPO API není nakonfigurováno)[/yellow]")

    # Check name
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Prohledávám databázi...", total=None)
        result = checker.check_name(nazev, nice_classes)

    # Display results
    console.print()
    risk_colors = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
    risk_emojis = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}

    console.print(Panel(
        f"[bold]Riziko: [{risk_colors[result.risk_level]}]{risk_emojis[result.risk_level]} "
        f"{result.risk_level}[/{risk_colors[result.risk_level]}][/bold]",
        title=f"Trademark Clearance – {nazev}",
    ))

    if result.overlapping_classes:
        console.print(f"[bold]Překrývající se třídy:[/bold] {', '.join(map(str, result.overlapping_classes))}")

    console.print(f"\n[bold]Doporučení:[/bold] {result.recommendation}")

    if result.conflicts:
        console.print(f"\n[bold]Nalezené konflikty ({len(result.conflicts)}):[/bold]")

        table = Table()
        table.add_column("Známka", style="cyan")
        table.add_column("Číslo")
        table.add_column("Status")
        table.add_column("Třídy")
        table.add_column("Podobnost", justify="right")

        for c in result.conflicts[:15]:
            sim_color = "red" if c.similarity_score > 0.8 else ("yellow" if c.similarity_score > 0.6 else "dim")
            table.add_row(
                c.name,
                c.application_number,
                c.status,
                ", ".join(map(str, c.nice_classes[:5])),
                f"[{sim_color}]{c.similarity_percent}%[/{sim_color}]",
            )

        console.print(table)

        if len(result.conflicts) > 15:
            console.print(f"[dim]... a dalších {len(result.conflicts) - 15} konfliktů[/dim]")
    else:
        console.print("\n[green]Žádné konfliktní známky nenalezeny.[/green]")

    # Save to database if candidate exists
    repo = CandidateRepository(settings)
    if repo.exists(nazev):
        conflicts_data = [
            {
                "name": c.name,
                "application_number": c.application_number,
                "status": c.status,
                "nice_classes": c.nice_classes,
                "similarity_score": c.similarity_score,
            }
            for c in result.conflicts
        ]
        repo.save_trademark_result(nazev, result.risk_level, conflicts_data)
        console.print(f"\n[dim]Výsledek uložen do databáze.[/dim]")


@cli.command("clearance-batch")
@click.argument("soubor", type=click.Path(exists=True))
@click.option(
    "--tridy",
    "-t",
    type=str,
    default="",
    help="Seznam tříd Nice oddělených čárkou (např. 9,35,42)",
)
@click.option(
    "--vystup",
    "-o",
    type=click.Path(),
    default=None,
    help="Soubor pro export výsledků (CSV)",
)
@click.pass_context
def check_trademark_batch(
    ctx: click.Context,
    soubor: str,
    tridy: str,
    vystup: str | None,
) -> None:
    """Hromadně zkontroluj názvy ze souboru.

    Načte názvy ze souboru (jeden na řádek) a zkontroluje
    každý proti databázi ochranných známek.

    Příklady:
        brand-gen clearance-batch nazvy.txt
        brand-gen clearance-batch nazvy.txt --tridy 9,35,42 --vystup report.csv
    """
    settings = ctx.obj["settings"]

    # Parse Nice classes
    if tridy:
        nice_classes = [int(c.strip()) for c in tridy.split(",") if c.strip()]
    else:
        nice_classes = settings.trademark.default_nice_classes

    # Load names from file
    with open(soubor, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    if not names:
        console.print("[red]Soubor neobsahuje žádné názvy.[/red]")
        return

    console.print(f"\n[bold]Hromadná kontrola ochranných známek[/bold]")
    console.print(f"[dim]Počet názvů: {len(names)}[/dim]")
    console.print(f"[dim]Třídy Nice: {', '.join(map(str, nice_classes))}[/dim]")

    # Create checker - use env vars as fallback if not in config
    import os
    client_id = settings.trademark.euipo.client_id or os.getenv("EUIPO_CLIENT_ID")
    client_secret = settings.trademark.euipo.client_secret or os.getenv("EUIPO_CLIENT_SECRET")

    checker = create_checker_from_settings(
        client_id=client_id,
        client_secret=client_secret,
        sandbox=settings.trademark.euipo.sandbox,
        similarity_threshold=settings.trademark.similarity_threshold,
    )

    if checker.is_using_mock:
        console.print("[yellow]⚠ Používám simulovaná data (EUIPO API není nakonfigurováno)[/yellow]")

    # Check all names
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Kontroluji názvy...", total=len(names))

        for name in names:
            result = checker.check_name(name, nice_classes)
            results.append(result)
            progress.update(task, advance=1, description=f"Kontroluji: {name}")

    # Count results
    high_count = sum(1 for r in results if r.risk_level == "HIGH")
    medium_count = sum(1 for r in results if r.risk_level == "MEDIUM")
    low_count = sum(1 for r in results if r.risk_level == "LOW")

    # Summary
    console.print(f"\n[bold]Souhrn:[/bold]")
    console.print(f"  🔴 Vysoké riziko: [red]{high_count}[/red]")
    console.print(f"  🟠 Střední riziko: [yellow]{medium_count}[/yellow]")
    console.print(f"  🟢 Nízké riziko: [green]{low_count}[/green]")

    # Display table
    table = Table(title="Výsledky kontroly")
    table.add_column("Název", style="cyan")
    table.add_column("Riziko", justify="center")
    table.add_column("Konflikty", justify="right")
    table.add_column("Max. podobnost", justify="right")

    for r in sorted(results, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x.risk_level]):
        risk_colors = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
        risk_emojis = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
        table.add_row(
            r.candidate_name,
            f"[{risk_colors[r.risk_level]}]{risk_emojis[r.risk_level]} {r.risk_level}[/{risk_colors[r.risk_level]}]",
            str(r.conflict_count),
            f"{r.max_similarity * 100:.0f}%" if r.conflicts else "-",
        )

    console.print(table)

    # Export to CSV
    if vystup:
        with open(vystup, "w", encoding="utf-8") as f:
            f.write("name,risk_level,conflict_count,max_similarity,overlapping_classes\n")
            for r in results:
                f.write(
                    f"{r.candidate_name},{r.risk_level},{r.conflict_count},"
                    f"{r.max_similarity:.2f},\"{','.join(map(str, r.overlapping_classes))}\"\n"
                )
        console.print(f"\n[green]Export uložen do: {vystup}[/green]")

    # Safe candidates
    safe = [r.candidate_name for r in results if r.risk_level == "LOW"]
    if safe:
        console.print(f"\n[bold green]Bezpečné názvy ({len(safe)}):[/bold green]")
        for name in safe[:10]:
            console.print(f"  ✓ {name}")
        if len(safe) > 10:
            console.print(f"  [dim]... a dalších {len(safe) - 10}[/dim]")


@cli.command("trademark-stats")
@click.pass_context
def show_trademark_stats(ctx: click.Context) -> None:
    """Zobraz statistiky kontroly ochranných známek."""
    settings = ctx.obj["settings"]
    repo = CandidateRepository(settings)

    stats = repo.count_by_trademark_risk()

    console.print("\n[bold]Statistiky kontroly ochranných známek[/bold]")
    console.print(f"  🔴 Vysoké riziko: [red]{stats['HIGH']}[/red]")
    console.print(f"  🟠 Střední riziko: [yellow]{stats['MEDIUM']}[/yellow]")
    console.print(f"  🟢 Nízké riziko: [green]{stats['LOW']}[/green]")
    console.print(f"  ⏳ Nezkontrolováno: [dim]{stats['unchecked']}[/dim]")

    total_checked = stats["HIGH"] + stats["MEDIUM"] + stats["LOW"]
    if total_checked > 0:
        safe_pct = stats["LOW"] / total_checked * 100
        console.print(f"\n[dim]Bezpečných kandidátů: {safe_pct:.1f}%[/dim]")


@cli.command("trademark-safe")
@click.option("--limit", "-l", default=20, help="Počet zobrazených kandidátů")
@click.pass_context
def show_trademark_safe(ctx: click.Context, limit: int) -> None:
    """Zobraz kandidáty s nízkým rizikem kolize ochranných známek."""
    settings = ctx.obj["settings"]
    repo = CandidateRepository(settings)

    candidates = repo.get_trademark_safe(limit=limit)

    if not candidates:
        console.print("[yellow]Žádní kandidáti s nízkým rizikem zatím nejsou.[/yellow]")
        console.print("[dim]Tip: Použij 'brand-gen clearance NAZEV' pro kontrolu.[/dim]")
        return

    table = Table(title=f"Kandidáti s nízkým rizikem kolize ({len(candidates)})")
    table.add_column("Název", style="bold cyan")
    table.add_column("LLM Skóre", justify="right")
    table.add_column("Heur. Skóre", justify="right")
    table.add_column("Zkontrolováno")

    for c in candidates:
        llm_str = str(c.llm_score) if c.llm_score else "-"
        heur_str = f"{c.heuristic_score:.1f}" if c.heuristic_score else "-"
        checked_str = c.trademark_checked_at.strftime("%Y-%m-%d") if c.trademark_checked_at else "-"
        table.add_row(c.name, f"[green]{llm_str}[/green]", heur_str, checked_str)

    console.print(table)


@cli.command("modely")
def show_models_info() -> None:
    """Zobraz informace o použitých matematických modelech."""
    console.print("\n[bold]📊 Matematické modely pro hodnocení názvů[/bold]\n")

    models = [
        (
            "Markovovy řetězce (Bigram/Trigram)",
            "Měří přirozenost sekvencí písmen na základě pravděpodobnostního modelu. "
            "Využívá frekvenční data z přirozených jazyků.",
            "P(word) = Π P(char_i | char_{i-1})",
        ),
        (
            "Sonority Sequencing Principle (SSP)",
            "Hodnotí vyslovitelnost na základě zvukové struktury slabik. "
            "Ideální slabika stoupá v sonorite k jádru (samohlásce) a pak klesá.",
            "Sonority scale: Stops(1) < Fricatives(3) < Nasals(5) < Liquids(6) < Vowels(10)",
        ),
        (
            "Shannon Entropy",
            "Měří informační obsah a předvídatelnost. Optimální názvy mají střední entropii - "
            "nejsou příliš náhodné ani příliš opakující se.",
            "H(word) = -Σ p(x) × log₂(p(x))",
        ),
        (
            "Phonotactic Probability",
            "Hodnotí, jak dobře název odpovídá pravidlům zvukových kombinací v jazyce. "
            "Zahrnuje pozici fonému (začátek, konec) i bifonové pravděpodobnosti.",
            "PP(word) = P(onset) × P(coda) × Π P(bigram_i)",
        ),
        (
            "Zipfův zákon",
            "Hodnotí použití běžných vs vzácných fonémů. Slova používající běžnější "
            "fonémy jsou typicky snáze vyslovitelná a zapamatovatelná.",
            "f(rank) ∝ 1/rank^α",
        ),
        (
            "Levenshtein Distance",
            "Měří unikátnost jako minimální editační vzdálenost od existujících slov a značek. "
            "Vyšší vzdálenost = unikátnější název.",
            "d(s1, s2) = min(insertions + deletions + substitutions)",
        ),
    ]

    for name, desc, formula in models:
        console.print(Panel(
            f"[dim]{desc}[/dim]\n\n[cyan]Vzorec:[/cyan] {formula}",
            title=f"[bold]{name}[/bold]",
            border_style="blue",
        ))
        console.print()


def main() -> None:
    """Entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()

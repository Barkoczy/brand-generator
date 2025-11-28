"""Genetic algorithm optimizer for brand name generation.

Uses evolutionary computation to optimize candidate generation:
- Selection: Tournament selection of fittest candidates
- Crossover: Combine parts of successful names
- Mutation: Random character mutations
- Fitness: Combined phonetic scoring
"""

import random
from dataclasses import dataclass, field
from typing import Callable

from src.config import Settings
from src.generator.candidate_generator import Candidate, CandidateGenerator
from src.scoring.phonetic_models import CombinedPhoneticScorer, PhoneticScoreResult


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""

    population_size: int = 200
    generations: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    tournament_size: int = 5
    elite_count: int = 10  # Top candidates preserved unchanged
    target_score: float = 8.0  # Stop if this score is achieved
    max_stagnation: int = 10  # Stop if no improvement for N generations


@dataclass
class Individual:
    """Individual in the genetic population."""

    genome: str  # The word as a string of characters
    pattern: str  # C/V pattern
    fitness: float = 0.0
    score_result: PhoneticScoreResult | None = None

    def to_candidate(self) -> Candidate:
        """Convert to Candidate object."""
        return Candidate(
            name=self.genome.capitalize(),
            pattern=self.pattern,
            raw_name=self.genome.lower(),
        )


@dataclass
class GenerationStats:
    """Statistics for a generation."""

    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    best_word: str
    diversity: float  # Measure of population diversity


class GeneticOptimizer:
    """Genetic algorithm optimizer for generating optimal brand names.

    The algorithm evolves a population of candidate names toward
    higher phonetic scores through selection, crossover, and mutation.
    """

    def __init__(
        self,
        settings: Settings,
        config: GeneticConfig | None = None,
        fitness_function: Callable[[str], float] | None = None,
        progress_callback: Callable[[GenerationStats], None] | None = None,
    ):
        """Initialize genetic optimizer.

        Args:
            settings: Application settings
            config: Genetic algorithm configuration
            fitness_function: Custom fitness function (defaults to phonetic scoring)
            progress_callback: Callback for progress updates
        """
        self.settings = settings
        self.config = config or GeneticConfig()
        self.progress_callback = progress_callback

        # Initialize helper components
        self.generator = CandidateGenerator(settings)
        self.scorer = CombinedPhoneticScorer()

        # Use custom fitness or default to combined phonetic score
        self.fitness_function = fitness_function or self._default_fitness

        # Character pools
        self.consonants = list(settings.consonants)
        self.vowels = list(settings.vowels)

    def _default_fitness(self, word: str) -> float:
        """Default fitness function using combined phonetic scoring."""
        result = self.scorer.score(word)
        return result.total_score

    def _get_pattern(self, word: str) -> str:
        """Determine C/V pattern of a word."""
        vowels = set(self.settings.vowels)
        return "".join("V" if c.lower() in vowels else "C" for c in word)

    def _random_char(self, char_type: str) -> str:
        """Get random character of specified type."""
        if char_type == "C":
            return random.choice(self.consonants)
        else:  # V
            return random.choice(self.vowels)

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize_population(self) -> list[Individual]:
        """Create initial population of random individuals.

        Returns:
            List of Individual objects
        """
        population = []

        for _ in range(self.config.population_size):
            # Generate random candidate using existing generator
            candidates = self.generator.generate(1)
            if candidates:
                candidate = list(candidates)[0]
                individual = Individual(
                    genome=candidate.raw_name,
                    pattern=candidate.pattern,
                )
                population.append(individual)

        return population

    def seed_population(self, seed_words: list[str]) -> list[Individual]:
        """Create initial population seeded with known good words.

        Args:
            seed_words: List of words to include in initial population

        Returns:
            List of Individual objects
        """
        population = []

        # Add seed words
        for word in seed_words[:self.config.population_size // 2]:
            if self.generator.passes_all_filters(word.lower()):
                individual = Individual(
                    genome=word.lower(),
                    pattern=self._get_pattern(word),
                )
                population.append(individual)

        # Fill rest with random individuals
        while len(population) < self.config.population_size:
            candidates = self.generator.generate(1)
            if candidates:
                candidate = list(candidates)[0]
                individual = Individual(
                    genome=candidate.raw_name,
                    pattern=candidate.pattern,
                )
                population.append(individual)

        return population

    # =========================================================================
    # FITNESS EVALUATION
    # =========================================================================

    def evaluate_fitness(self, population: list[Individual]) -> list[Individual]:
        """Evaluate fitness of all individuals in population.

        Args:
            population: List of individuals

        Returns:
            Same list with fitness values updated
        """
        for individual in population:
            individual.fitness = self.fitness_function(individual.genome)

            # Also store full score result for detailed analysis
            individual.score_result = self.scorer.score(individual.genome)

        return population

    # =========================================================================
    # SELECTION
    # =========================================================================

    def tournament_selection(self, population: list[Individual]) -> Individual:
        """Select individual using tournament selection.

        Args:
            population: Population to select from

        Returns:
            Selected individual
        """
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def select_parents(self, population: list[Individual]) -> list[Individual]:
        """Select parents for next generation.

        Args:
            population: Current population

        Returns:
            List of selected parents
        """
        parents = []
        for _ in range(len(population)):
            parent = self.tournament_selection(population)
            parents.append(parent)
        return parents

    # =========================================================================
    # CROSSOVER
    # =========================================================================

    def single_point_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual]:
        """Perform single-point crossover between two parents.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring individuals
        """
        if random.random() > self.config.crossover_rate:
            return parent1, parent2

        # Find crossover point
        min_len = min(len(parent1.genome), len(parent2.genome))
        if min_len <= 2:
            return parent1, parent2

        crossover_point = random.randint(1, min_len - 1)

        # Create offspring
        child1_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
        child2_genome = parent2.genome[:crossover_point] + parent1.genome[crossover_point:]

        # Validate and adjust length
        child1_genome = self._adjust_length(child1_genome)
        child2_genome = self._adjust_length(child2_genome)

        return (
            Individual(genome=child1_genome, pattern=self._get_pattern(child1_genome)),
            Individual(genome=child2_genome, pattern=self._get_pattern(child2_genome)),
        )

    def uniform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual]:
        """Perform uniform crossover between two parents.

        Each position has 50% chance of coming from either parent.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring individuals
        """
        if random.random() > self.config.crossover_rate:
            return parent1, parent2

        min_len = min(len(parent1.genome), len(parent2.genome))
        max_len = max(len(parent1.genome), len(parent2.genome))

        child1_chars = []
        child2_chars = []

        for i in range(max_len):
            if i < min_len:
                if random.random() < 0.5:
                    child1_chars.append(parent1.genome[i])
                    child2_chars.append(parent2.genome[i])
                else:
                    child1_chars.append(parent2.genome[i])
                    child2_chars.append(parent1.genome[i])
            elif i < len(parent1.genome):
                if random.random() < 0.5:
                    child1_chars.append(parent1.genome[i])
            elif i < len(parent2.genome):
                if random.random() < 0.5:
                    child2_chars.append(parent2.genome[i])

        child1_genome = self._adjust_length("".join(child1_chars))
        child2_genome = self._adjust_length("".join(child2_chars))

        return (
            Individual(genome=child1_genome, pattern=self._get_pattern(child1_genome)),
            Individual(genome=child2_genome, pattern=self._get_pattern(child2_genome)),
        )

    def _adjust_length(self, genome: str) -> str:
        """Adjust genome length to be within valid range."""
        min_len = self.settings.min_length
        max_len = self.settings.max_length

        # Truncate if too long
        if len(genome) > max_len:
            genome = genome[:max_len]

        # Extend if too short
        while len(genome) < min_len:
            # Add character that continues pattern
            pattern = self._get_pattern(genome)
            if pattern and pattern[-1] == "C":
                genome += random.choice(self.vowels)
            else:
                genome += random.choice(self.consonants)

        return genome

    # =========================================================================
    # MUTATION
    # =========================================================================

    def mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual.

        Mutation types:
        - Point mutation: Replace single character
        - Swap mutation: Swap two adjacent characters
        - Insert mutation: Insert new character
        - Delete mutation: Remove character

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        if random.random() > self.config.mutation_rate:
            return individual

        genome = list(individual.genome)
        mutation_type = random.choice(["point", "swap", "insert", "delete"])

        if mutation_type == "point" and genome:
            # Replace random character with same type (C or V)
            pos = random.randint(0, len(genome) - 1)
            current_type = "V" if genome[pos] in self.vowels else "C"
            genome[pos] = self._random_char(current_type)

        elif mutation_type == "swap" and len(genome) >= 2:
            # Swap two adjacent characters
            pos = random.randint(0, len(genome) - 2)
            genome[pos], genome[pos + 1] = genome[pos + 1], genome[pos]

        elif mutation_type == "insert" and len(genome) < self.settings.max_length:
            # Insert random character
            pos = random.randint(0, len(genome))
            # Determine type based on neighbors
            if pos == 0 or (pos > 0 and genome[pos - 1] in self.vowels):
                new_char = random.choice(self.consonants)
            else:
                new_char = random.choice(self.vowels)
            genome.insert(pos, new_char)

        elif mutation_type == "delete" and len(genome) > self.settings.min_length:
            # Delete random character
            pos = random.randint(0, len(genome) - 1)
            del genome[pos]

        new_genome = "".join(genome)

        # Ensure validity
        if not self.generator.passes_all_filters(new_genome):
            return individual  # Keep original if mutated version is invalid

        return Individual(
            genome=new_genome,
            pattern=self._get_pattern(new_genome),
        )

    def adaptive_mutate(self, individual: Individual, generation: int) -> Individual:
        """Apply adaptive mutation with rate depending on generation.

        Early generations: Higher mutation for exploration
        Later generations: Lower mutation for exploitation

        Args:
            individual: Individual to mutate
            generation: Current generation number

        Returns:
            Mutated individual
        """
        # Adaptive mutation rate decreases over generations
        progress = generation / self.config.generations
        adaptive_rate = self.config.mutation_rate * (1 - progress * 0.5)

        # Temporarily adjust mutation rate
        original_rate = self.config.mutation_rate
        self.config.mutation_rate = adaptive_rate
        result = self.mutate(individual)
        self.config.mutation_rate = original_rate

        return result

    # =========================================================================
    # DIVERSITY MAINTENANCE
    # =========================================================================

    def calculate_diversity(self, population: list[Individual]) -> float:
        """Calculate population diversity.

        Uses average pairwise Hamming distance.

        Args:
            population: Current population

        Returns:
            Diversity score (0-1)
        """
        if len(population) < 2:
            return 0.0

        total_distance = 0
        comparisons = 0

        # Sample pairs for efficiency
        sample_size = min(50, len(population))
        sample = random.sample(population, sample_size)

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                g1, g2 = sample[i].genome, sample[j].genome
                min_len = min(len(g1), len(g2))
                max_len = max(len(g1), len(g2))

                if max_len == 0:
                    continue

                # Hamming distance + length difference
                hamming = sum(g1[k] != g2[k] for k in range(min_len))
                total_distance += (hamming + (max_len - min_len)) / max_len
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0

    def inject_diversity(self, population: list[Individual], count: int) -> list[Individual]:
        """Inject new random individuals to maintain diversity.

        Args:
            population: Current population
            count: Number of new individuals to inject

        Returns:
            Population with new individuals
        """
        # Remove worst individuals
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        population = population[:-count]

        # Add new random individuals
        for _ in range(count):
            candidates = self.generator.generate(1)
            if candidates:
                candidate = list(candidates)[0]
                individual = Individual(
                    genome=candidate.raw_name,
                    pattern=candidate.pattern,
                )
                population.append(individual)

        return population

    # =========================================================================
    # MAIN EVOLUTION LOOP
    # =========================================================================

    def evolve(self, seed_words: list[str] | None = None) -> list[Individual]:
        """Run genetic algorithm evolution.

        Args:
            seed_words: Optional list of words to seed population

        Returns:
            Final population sorted by fitness
        """
        # Initialize population
        if seed_words:
            population = self.seed_population(seed_words)
        else:
            population = self.initialize_population()

        # Evaluate initial fitness
        population = self.evaluate_fitness(population)

        best_fitness_ever = 0.0
        stagnation_count = 0

        for generation in range(self.config.generations):
            # Sort by fitness
            population = sorted(population, key=lambda x: x.fitness, reverse=True)

            # Calculate statistics
            fitnesses = [ind.fitness for ind in population]
            stats = GenerationStats(
                generation=generation,
                best_fitness=max(fitnesses),
                avg_fitness=sum(fitnesses) / len(fitnesses),
                worst_fitness=min(fitnesses),
                best_word=population[0].genome,
                diversity=self.calculate_diversity(population),
            )

            # Progress callback
            if self.progress_callback:
                self.progress_callback(stats)

            # Check termination conditions
            if stats.best_fitness >= self.config.target_score:
                break  # Target achieved

            if stats.best_fitness > best_fitness_ever:
                best_fitness_ever = stats.best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= self.config.max_stagnation:
                # Inject diversity to escape local optimum
                population = self.inject_diversity(
                    population, self.config.population_size // 10
                )
                stagnation_count = 0

            # Elitism: Keep best individuals unchanged
            elite = population[:self.config.elite_count]

            # Selection
            parents = self.select_parents(population)

            # Create new population through crossover and mutation
            new_population = list(elite)  # Start with elite

            while len(new_population) < self.config.population_size:
                # Select two parents
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                # Crossover
                child1, child2 = self.single_point_crossover(parent1, parent2)

                # Mutation with adaptive rate
                child1 = self.adaptive_mutate(child1, generation)
                child2 = self.adaptive_mutate(child2, generation)

                # Validate offspring
                if self.generator.passes_all_filters(child1.genome):
                    new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    if self.generator.passes_all_filters(child2.genome):
                        new_population.append(child2)

            # Evaluate fitness of new population
            population = self.evaluate_fitness(new_population)

        # Final sort
        population = sorted(population, key=lambda x: x.fitness, reverse=True)

        return population

    def get_best_candidates(
        self, count: int = 10, seed_words: list[str] | None = None
    ) -> list[Candidate]:
        """Run evolution and return best candidates.

        Args:
            count: Number of top candidates to return
            seed_words: Optional seed words for initial population

        Returns:
            List of best Candidate objects
        """
        population = self.evolve(seed_words)
        top_individuals = population[:count]
        return [ind.to_candidate() for ind in top_individuals]


# =============================================================================
# ISLAND MODEL (PARALLEL EVOLUTION)
# =============================================================================

@dataclass
class IslandConfig:
    """Configuration for island model."""

    num_islands: int = 4
    migration_rate: float = 0.1
    migration_interval: int = 10  # Generations between migrations


class IslandOptimizer:
    """Island model genetic algorithm for increased diversity.

    Runs multiple independent populations (islands) with periodic
    migration of best individuals between islands.
    """

    def __init__(
        self,
        settings: Settings,
        genetic_config: GeneticConfig | None = None,
        island_config: IslandConfig | None = None,
        progress_callback: Callable[[int, list[GenerationStats]], None] | None = None,
    ):
        """Initialize island optimizer.

        Args:
            settings: Application settings
            genetic_config: Configuration for each island's GA
            island_config: Island model configuration
            progress_callback: Callback for progress (island_id, stats_list)
        """
        self.settings = settings
        self.genetic_config = genetic_config or GeneticConfig()
        self.island_config = island_config or IslandConfig()
        self.progress_callback = progress_callback

        # Create optimizers for each island
        self.islands = [
            GeneticOptimizer(settings, self.genetic_config)
            for _ in range(self.island_config.num_islands)
        ]

    def migrate(self, populations: list[list[Individual]]) -> list[list[Individual]]:
        """Perform migration between islands.

        Best individuals from each island migrate to next island (ring topology).

        Args:
            populations: List of population lists (one per island)

        Returns:
            Updated populations after migration
        """
        num_migrants = int(
            self.genetic_config.population_size * self.island_config.migration_rate
        )

        migrants = []
        for pop in populations:
            # Sort and get best individuals
            sorted_pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
            migrants.append(sorted_pop[:num_migrants])

        # Ring migration: island i receives migrants from island i-1
        for i, pop in enumerate(populations):
            source_island = (i - 1) % len(populations)
            # Replace worst individuals with migrants
            pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
            pop = pop[:-num_migrants] + migrants[source_island]
            populations[i] = pop

        return populations

    def evolve(self, seed_words: list[str] | None = None) -> list[Individual]:
        """Run island model evolution.

        Args:
            seed_words: Optional seed words

        Returns:
            Combined best individuals from all islands
        """
        # Initialize all island populations
        populations = []
        for island in self.islands:
            if seed_words:
                pop = island.seed_population(seed_words)
            else:
                pop = island.initialize_population()
            pop = island.evaluate_fitness(pop)
            populations.append(pop)

        # Evolve for specified generations
        for gen in range(self.genetic_config.generations):
            # Evolve each island for one generation
            for i, (island, pop) in enumerate(zip(self.islands, populations)):
                # Selection
                parents = island.select_parents(pop)

                # Elitism
                elite = sorted(pop, key=lambda x: x.fitness, reverse=True)
                elite = elite[:self.genetic_config.elite_count]

                # Create new population
                new_pop = list(elite)
                while len(new_pop) < self.genetic_config.population_size:
                    p1, p2 = random.choice(parents), random.choice(parents)
                    c1, c2 = island.single_point_crossover(p1, p2)
                    c1 = island.adaptive_mutate(c1, gen)
                    c2 = island.adaptive_mutate(c2, gen)

                    if island.generator.passes_all_filters(c1.genome):
                        new_pop.append(c1)
                    if len(new_pop) < self.genetic_config.population_size:
                        if island.generator.passes_all_filters(c2.genome):
                            new_pop.append(c2)

                populations[i] = island.evaluate_fitness(new_pop)

            # Periodic migration
            if gen > 0 and gen % self.island_config.migration_interval == 0:
                populations = self.migrate(populations)

        # Combine all populations and return best
        all_individuals = []
        for pop in populations:
            all_individuals.extend(pop)

        # Remove duplicates
        seen = set()
        unique = []
        for ind in all_individuals:
            if ind.genome not in seen:
                seen.add(ind.genome)
                unique.append(ind)

        return sorted(unique, key=lambda x: x.fitness, reverse=True)

    def get_best_candidates(
        self, count: int = 10, seed_words: list[str] | None = None
    ) -> list[Candidate]:
        """Run island evolution and return best candidates.

        Args:
            count: Number of top candidates
            seed_words: Optional seed words

        Returns:
            List of best Candidate objects
        """
        population = self.evolve(seed_words)
        return [ind.to_candidate() for ind in population[:count]]

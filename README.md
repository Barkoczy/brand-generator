# Brand Generator

An intelligent neologism generator for brand names with multi-provider LLM support and advanced mathematical models. Generates invented words (coined terms) suitable for trademark registration.

## Features

- **C/V Pattern Generation**: Creates pronounceable neologisms using consonant-vowel patterns
- **Advanced Phonetic Scoring**: Mathematical models for evaluating name quality
- **Genetic Algorithm**: Evolutionary optimization for finding optimal names
- **LLM Evaluation**: Intelligent semantic evaluation using AI models
- **Multi-Provider Support**: Works with Anthropic, OpenAI, Google Gemini, Ollama, and LM Studio
- **Trademark-Focused**: Filters out descriptive terms that are hard to register
- **SQLite Storage**: Persistent storage for candidates with full evaluation history
- **Czech CLI**: Command-line interface in Czech language

## Mathematical Models

The generator uses six advanced mathematical models for phonetic evaluation:

| Model | Description | Formula |
|-------|-------------|---------|
| **Markov Chains** | Natural phoneme sequence probability | `P(word) = Π P(char_i \| char_{i-1})` |
| **Sonority (SSP)** | Syllable structure evaluation | Stops(1) < Fricatives(3) < Nasals(5) < Liquids(6) < Vowels(10) |
| **Shannon Entropy** | Pattern balance measurement | `H = -Σ p(x) × log₂(p(x))` |
| **Phonotactics** | Sound combination probability | `PP = P(onset) × P(coda) × Π P(bigram)` |
| **Zipf's Law** | Phoneme frequency weighting | `f(rank) ∝ 1/rank^α` |
| **Levenshtein** | Uniqueness via edit distance | `d(s1,s2) = min(ins + del + sub)` |

## Installation

```bash
# Clone the repository
git clone https://github.com/Barkoczy/brand-generator.git
cd brand-generator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with your preferred LLM provider
pip install -e ".[gemini]"     # Google Gemini (recommended)
pip install -e ".[anthropic]"  # Anthropic Claude
pip install -e ".[openai]"     # OpenAI GPT
pip install -e ".[all]"        # All cloud providers
pip install -e "."             # Base only (Ollama/LM Studio via HTTP)
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

```env
# Google Gemini (recommended)
GOOGLE_API_KEY=your-api-key

# Anthropic Claude
ANTHROPIC_API_KEY=your-api-key

# OpenAI
OPENAI_API_KEY=your-api-key
```

### Configuration File

Edit `config.yaml` to customize the generator:

```yaml
llm:
  provider: gemini # anthropic, openai, ollama, lmstudio, gemini
  model: gemini-2.0-flash
  batch_size: 20
  temperature: 0.3

# Banned substrings (makes names descriptive)
banned_substrings:
  - doc
  - file
  - cloud
  - data
  # ... see config.yaml for full list

# C/V patterns for name generation
patterns:
  - CVCV # 4 chars: Nalo
  - CVCVCV # 6 chars: Nalovi
  - CVCCVC # 6 chars: Narvos

# Genetic algorithm settings
genetic:
  population_size: 200
  generations: 50
  mutation_rate: 0.15
  target_score: 8.0

# Phonetic model weights (must sum to 1.0)
phonetic_weights:
  markov: 0.20
  sonority: 0.20
  entropy: 0.10
  phonotactic: 0.20
  zipf: 0.10
  uniqueness: 0.20
```

## Usage

### Basic Commands

```bash
# Check available providers
brand-gen providery

# Show mathematical models info
brand-gen modely
```

### Generate Names (Offline Mode)

Generate candidates using heuristic + phonetic scoring (no LLM required):

```bash
brand-gen generuj --pocet 100
```

### Genetic Algorithm (Recommended)

Use evolutionary optimization for best results:

```bash
# Basic evolution
brand-gen evoluce --populace 200 --generace 50

# With custom target score
brand-gen evoluce --cil 9.0 --generace 100

# Island model (parallel evolution)
brand-gen evoluce --ostrovy 4 --populace 500

# Export results
brand-gen evoluce --vystup best_names.txt
```

### Phonetic Analysis

Analyze any name with detailed phonetic scoring:

```bash
# Analyze single name
brand-gen analyzuj Valujo

# Compare multiple names
brand-gen porovnej Valujo Nextra Zenira Google Apple
```

Example output:
```
┌─────────────────────── Fonetická analýza ────────────────────────┐
│ Valujo                                                           │
└──────────────────────────────────────────────────────────────────┘

Celkové skóre: 7.12/10

┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Model            ┃ Skóre ┃ Váha ┃ Příspěvek ┃ Popis              ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Markov (n-gramy) │   4.2 │  20% │      0.84 │ Přirozenost sekvencí│
│ Sonorita (SSP)   │  10.0 │  20% │      2.00 │ Slabičná struktura │
│ Entropie         │  10.0 │  10% │      1.00 │ Vyváženost vzorů   │
│ Fonotaktika      │   6.0 │  20% │      1.19 │ Zvukové kombinace  │
│ Zipfův zákon     │   3.9 │  10% │      0.39 │ Frekvence fonémů   │
│ Unikátnost       │   8.5 │  20% │      1.70 │ Editační vzdálenost│
└──────────────────┴───────┴──────┴───────────┴────────────────────┘
```

### Run Autonomous Agent (LLM Mode)

Generate and evaluate names using AI:

```bash
# Use default provider from config.yaml
brand-gen spust --cil 10 --iterace 5

# Override provider via CLI
brand-gen spust --provider gemini --model gemini-2.0-flash --cil 10
brand-gen spust --provider ollama --model llama3.2 --cil 5
```

### View Results

```bash
# Show top candidates
brand-gen top --limit 20

# Show candidate details
brand-gen detail Valujo

# Show database statistics
brand-gen statistiky
```

### Manage Candidates

```bash
# Mark as favorite
brand-gen oblibene Valujo

# Reject candidate
brand-gen vyrad BadName
```

### Export

```bash
brand-gen export names.txt --min-skore 7
brand-gen export names.csv --format csv
brand-gen export names.json --format json
```

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `generuj` | Generate names with heuristic scoring (offline) |
| `evoluce` | Run genetic algorithm optimization |
| `spust` | Run autonomous agent with LLM evaluation |
| `analyzuj <name>` | Detailed phonetic analysis of a name |
| `porovnej <names...>` | Compare multiple names side by side |
| `modely` | Show information about mathematical models |
| `providery` | List available LLM providers |
| `top` | Show top candidates from database |
| `detail <name>` | Show detailed info about a candidate |
| `statistiky` | Show database statistics |
| `oblibene <name>` | Mark candidate as favorite |
| `vyrad <name>` | Reject a candidate |
| `export <file>` | Export candidates to file |

## Supported LLM Providers

| Provider    | Model (default)          | Requirement                        |
| ----------- | ------------------------ | ---------------------------------- |
| `gemini`    | gemini-2.0-flash         | `GOOGLE_API_KEY`                   |
| `anthropic` | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY`                |
| `openai`    | gpt-4o                   | `OPENAI_API_KEY`                   |
| `ollama`    | llama3.2                 | Ollama server on `localhost:11434` |
| `lmstudio`  | local-model              | LM Studio on `localhost:1234/v1`   |

## Project Structure

```
brand-generator/
├── config.yaml              # Main configuration
├── .env                     # API keys (not in git)
├── .env.example             # Template for .env
├── prompts/
│   └── brand_evaluator_system.txt  # LLM system prompt
├── src/
│   ├── cli.py               # CLI interface
│   ├── config/              # Configuration loading
│   ├── generator/
│   │   ├── candidate_generator.py  # C/V pattern generation
│   │   └── genetic_optimizer.py    # Genetic algorithm
│   ├── scoring/
│   │   ├── heuristic_scorer.py     # Basic heuristics
│   │   └── phonetic_models.py      # Advanced math models
│   ├── llm/                 # LLM providers
│   │   ├── base.py          # Abstract provider
│   │   ├── factory.py       # Provider factory
│   │   ├── scorer.py        # Universal LLM scorer
│   │   └── providers/       # Provider implementations
│   ├── db/                  # SQLite repository
│   └── agent/               # Autonomous orchestrator
└── data/
    └── candidates.db        # SQLite database
```

## How It Works

### Generation Pipeline

1. **Pattern Generation**: Creates names using C/V patterns like `CVCV`, `CVCVCV`
2. **Filtering**: Removes names containing banned substrings (doc, file, cloud, etc.)
3. **Phonetic Scoring**: Evaluates using 6 mathematical models:
   - Markov chains for natural sequences
   - SSP for pronunciation ease
   - Entropy for pattern balance
   - Phonotactics for sound rules
   - Zipf's law for phoneme frequency
   - Levenshtein for uniqueness
4. **Genetic Optimization** (optional): Evolves population toward higher scores
5. **LLM Evaluation** (optional): AI rates trademark potential and brand fit
6. **Storage**: Saves candidates with scores to SQLite

### Genetic Algorithm

The genetic optimizer uses evolutionary computation:

- **Selection**: Tournament selection of fittest candidates
- **Crossover**: Single-point and uniform crossover operators
- **Mutation**: Point, swap, insert, and delete mutations
- **Elitism**: Top candidates preserved unchanged
- **Island Model**: Parallel populations with migration for diversity

## Trademark Considerations

The generator focuses on creating **coined/fanciful marks** - completely invented words with no dictionary meaning. These have the strongest trademark protection because they:

- Are inherently distinctive
- Cannot be claimed by competitors
- Are easier to register internationally

The system explicitly filters out:

- Descriptive terms (doc, file, cloud, data)
- Generic tech suffixes (-soft, -ware, -tech, -hub)
- Terms that suggest the product's function

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

# Brand Generator

An intelligent neologism generator for brand names with multi-provider LLM support. Generates invented words (coined terms) suitable for trademark registration.

## Features

- **C/V Pattern Generation**: Creates pronounceable neologisms using consonant-vowel patterns
- **Heuristic Scoring**: Fast pre-filtering based on phonetic rules
- **LLM Evaluation**: Intelligent semantic evaluation using AI models
- **Multi-Provider Support**: Works with Anthropic, OpenAI, Google Gemini, Ollama, and LM Studio
- **Trademark-Focused**: Filters out descriptive terms that are hard to register
- **SQLite Storage**: Persistent storage for candidates with full evaluation history
- **Czech CLI**: Command-line interface in Czech language

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
```

## Usage

### Check Available Providers

```bash
brand-gen providery
```

### Generate Names (Offline Mode)

Generate candidates using heuristic scoring only (no LLM required):

```bash
brand-gen generuj --pocet 100
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
│   ├── generator/           # Name generation logic
│   ├── scoring/             # Heuristic scoring
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

1. **Generation**: Creates names using C/V (consonant/vowel) patterns like `CVCV`, `CVCVCV`
2. **Filtering**: Removes names containing banned substrings (doc, file, cloud, etc.)
3. **Heuristic Scoring**: Evaluates pronounceability, visual balance, memorability
4. **LLM Evaluation**: AI rates names on trademark potential, uniqueness, brand fit
5. **Storage**: Saves all candidates with scores and recommendations to SQLite

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

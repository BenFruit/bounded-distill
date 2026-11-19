# Contributing to bounded-distill

Thanks for your interest! Bug reports, ideas and pull requests are welcome.

## Before you start

- Small fixes (typos, docs, an obvious bug): just open a pull request.
- Larger changes (new features, API changes, new dependencies): please open an issue first,
  so we can agree on the approach before you spend time on it.

The project values simplicity: small pure functions, no speculative abstractions, few
dependencies. The reasons behind the current design are in [docs/design.md](docs/design.md).

## Development setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/BenFruit/bounded-distill
cd bounded-distill
uv sync
```

## Checks

Every pull request must pass the same checks as CI:

```bash
uv run ruff check
uv run ruff format --check
uv run pyright          # strict mode
uv run pytest           # slow tests with the real embedder: uv run pytest -m slow
```

CI runs them on the oldest and the newest supported Python (3.10 and 3.14).

## Code guidelines

- Strict typing: the code must pass `pyright` in strict mode without new `# pyright: ignore`
  comments, except at the boundaries to untyped libraries.
- Every public function needs tests. Tests must not download models: use the fake embedder in
  `tests/synthetic.py`; anything that needs the network is marked `@pytest.mark.slow`.
- Code, comments and docs are in English.
- Keep pull requests focused: one change per pull request.
- Update the README and add a line to `CHANGELOG.md` when users would notice the change.

## Benchmarks

`benchmarks/` holds the scripts behind the benchmark in the README; see
[benchmarks/README.md](benchmarks/README.md). They need an OpenAI-compatible LLM endpoint and are
not required for most contributions. Never commit API keys, `.env` files or private data.

## Review and merge

The maintainer reviews every pull request and decides whether and when to merge it. CI must pass
and at least one approving review is required. Releases are made by the maintainer.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the license of this project.

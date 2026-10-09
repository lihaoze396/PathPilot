# Fuzzer

Research fuzzer for smart-contract logic bugs. The search core is derived from CrossFuzz. The instrumentation layer adds three observation stages:

- GEI: global exploration
- KSI: runtime-state evidence
- TFI: target-focused follow-up

Upstream license notice: `NOTICE_CROSSFUZZ_MIT.txt`.

## Layout

- `fuzzer/`: fuzzing engine, EVM adapter, GEI/KSI/TFI instrumentation, and optional LLM agent
- `configs/`: example search configs
- `examples/`: example path plan

Experiment outputs, datasets, API keys, and machine-specific launch scripts are not part of this release.

## Install

```powershell
python -m pip install -r requirements.txt
```

Solidity compilation also needs a local `solc`. Pass its directory with `--solc-dir` or set `SOLC_DIR`. Do not commit a compiler cache.

LLM guidance is optional. Copy `.env.llm.example` to a private `.env` outside version control and fill in your own key. The example file contains placeholders only.

## Run

From this directory:

```powershell
python -m fuzzer.main --source path\to\Contract.sol --contract ContractName --solc-dir path\to\solc --time-budget 60 --config configs\default.yaml --out out\demo
```

Useful flags:

- `--enable-stage`
- `--enable-gei-side-path`
- `--enable-ksi`
- `--enable-tfi-focus`

Run `python -m fuzzer.main --help` for the full list.

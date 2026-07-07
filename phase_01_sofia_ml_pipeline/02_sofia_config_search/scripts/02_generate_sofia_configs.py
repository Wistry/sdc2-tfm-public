"""Genera las configuraciones SoFiA de la fase 01.

La idea es partir de un archivo base común y producir variantes comparables
que representen decisiones de diseño distintas: más recall, más reliability,
o compromisos intermedios para alimentar la fase ML posterior.
"""

from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = DEMO_DIR / "configs"
BASE_CONFIG =  CONFIG_DIR / "test_dev_medium.par"


# Las variantes se escriben a partir de una base común. 
VARIANTS = {
    "baseline_current": {
        "input.data": "<FITS_PATH>",
        "output.directory": "<OUTPUT_DIR>",
        "output.filename": "baseline_current",
    },
    "sofia2_default_template": {
        "input.data": "<FITS_PATH>",
        "output.directory": "<OUTPUT_DIR>",
        "output.filename": "sofia2_default_template",
        "flag.threshold": "5.0",
        "scfind.kernelsXY": "0, 3, 6",
        "scfind.kernelsZ": "0, 3, 7, 15",
        "scfind.threshold": "5.0",
        "linker.radiusXY": "1",
        "linker.radiusZ": "1",
        "linker.minSizeXY": "5",
        "linker.minSizeZ": "5",
        "filter.discardNegative": "true",
        "reliability.enable": "false",
        "reliability.threshold": "0.9",
    },
    "sdc2_team_sofia_like": {
        "input.data": "<FITS_PATH>",
        "output.directory": "<OUTPUT_DIR>",
        "output.filename": "sdc2_team_sofia_like",
        "scfind.kernelsXY": "0, 3, 6",
        "scfind.kernelsZ": "0, 3, 7, 15, 31",
        "scfind.threshold": "3.8",
        "linker.radiusXY": "2",
        "linker.radiusZ": "2",
        "linker.minSizeXY": "3",
        "linker.minSizeZ": "3",
        "filter.discardNegative": "true",
        "reliability.enable": "true",
        "reliability.threshold": "0.1",
        "reliability.minSNR": "1.5",
        "reliability.scaleKernel": "0.3",
    },
    "hi_friends_dev12_like": {
        "input.data": "<FITS_PATH>",
        "output.directory": "<OUTPUT_DIR>",
        "output.filename": "hi_friends_dev12_like",
        "flag.threshold": "3.5",
        "scaleNoise.windowXY": "99",
        "scaleNoise.windowZ": "5",
        "scfind.kernelsXY": "0, 4, 8",
        "scfind.kernelsZ": "0, 5, 11, 21, 41",
        "scfind.threshold": "3.5",
        "scfind.replacement": "2.0",
        "linker.radiusXY": "4",
        "linker.radiusZ": "5",
        "linker.minSizeXY": "5",
        "linker.minSizeZ": "3",
        "filter.discardNegative": "true",
        "reliability.enable": "true",
        "reliability.threshold": "0.4",
        "filter.minSNR": "6.0",
        "reliability.scaleKernel": "0.3",
    },
    "hi_friends_yaml_like": {
        "input.data": "<FITS_PATH>",
        "output.directory": "<OUTPUT_DIR>",
        "output.filename": "hi_friends_yaml_like",
        "flag.threshold": "3.5",
        "scaleNoise.windowXY": "99",
        "scaleNoise.windowZ": "5",
        "scfind.kernelsXY": "0, 4, 8",
        "scfind.kernelsZ": "0, 5, 11, 21, 41",
        "scfind.threshold": "4.0",
        "scfind.replacement": "2.0",
        "linker.radiusXY": "4",
        "linker.radiusZ": "5",
        "linker.minSizeXY": "5",
        "linker.minSizeZ": "3",
        "filter.discardNegative": "true",
        "reliability.enable": "true",
        "reliability.threshold": "0.4",
        "filter.minSNR": "6.0",
        "reliability.scaleKernel": "0.3",
    },
    "loose_recall": {
        "input.data": "<FITS_PATH>",
        "output.directory": "<OUTPUT_DIR>",
        "output.filename": "loose_recall",
        "flag.threshold": "2.5",
        "scfind.kernelsXY": "0, 3, 6",
        "scfind.kernelsZ": "0, 3, 7, 15, 31",
        "scfind.threshold": "2.8",
        "linker.radiusXY": "3",
        "linker.radiusZ": "3",
        "linker.minSizeXY": "2",
        "linker.minSizeZ": "2",
        "filter.discardNegative": "false",
        "reliability.enable": "false",
    },
    "strict_reliability": {
        "input.data": "<FITS_PATH>",
        "output.directory": "<OUTPUT_DIR>",
        "output.filename": "strict_reliability",
        "flag.threshold": "3.2",
        "scfind.kernelsXY": "0, 3, 6",
        "scfind.kernelsZ": "0, 3, 7, 15",
        "scfind.threshold": "4.5",
        "linker.radiusXY": "1",
        "linker.radiusZ": "1",
        "linker.minSizeXY": "5",
        "linker.minSizeZ": "5",
        "filter.discardNegative": "true",
        "reliability.enable": "true",
        "reliability.threshold": "0.6",
        "reliability.minSNR": "2.0",
        "reliability.scaleKernel": "0.3",
    },
}


def read_base_config() -> list[str]:
    if not BASE_CONFIG.exists():
        raise SystemExit(f"Falta config base: {BASE_CONFIG}")
    return BASE_CONFIG.read_text(encoding="utf-8").splitlines()


def set_parameter(lines: list[str], key: str, value: str) -> list[str]:
    prefix = f"{key}"
    replaced = False
    output = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix) and "=" in stripped:
            output.append(f"{key:<27} = {value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{key:<27} = {value}")
    return output


def build_variant(base_lines: list[str], updates: dict[str, str]) -> str:
    lines = list(base_lines)
    for key, value in updates.items():
        lines = set_parameter(lines, key, value)
    return "\n".join(lines) + "\n"


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    base_lines = read_base_config()
    for name, updates in VARIANTS.items():
        path = CONFIG_DIR / f"{name}.par"
        path.write_text(build_variant(base_lines, updates), encoding="utf-8")
        print(f"Config escrita: {path}")


if __name__ == "__main__":
    main()

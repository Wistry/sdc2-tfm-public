"""Prepara un script con los comandos de ejecución de SoFiA.

Este archivo no ejecuta nada por sí mismo. Solo materializa un bash script
reproducible para lanzar una tanda de configuraciones .par ya existentes.
"""

from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = DEMO_DIR / "configs"
OUTPUTS_DIR = DEMO_DIR / "outputs"
CATALOGS_DIR = OUTPUTS_DIR / "catalogs"
RUN_COMMANDS = OUTPUTS_DIR / "run_commands.sh"


def main() -> None:
    # La fuente de verdad son los .par presentes en configs/.
    configs = sorted(CONFIG_DIR.glob("*.par"))
    if not configs:
        raise SystemExit(f"No hay configs .par en {CONFIG_DIR}. Ejecuta 02_generate_sofia_configs.py.")

    CATALOGS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        '# Placeholders:',
        'SOFIA_BIN="${SOFIA_BIN:-/path/to/sofia}"',
        'FITS_PATH="${FITS_PATH:-/path/to/subcube.fits}"',
        f'OUTPUT_DIR="{CATALOGS_DIR.resolve()}"',
        "",
        'mkdir -p "$OUTPUT_DIR"',
        "",
    ]
    for config in configs:
        run_config = CATALOGS_DIR / f"{config.stem}.run.par"
        lines.extend([
            f'CONFIG="{config.resolve()}"',
            f'RUN_CONFIG="{run_config.resolve()}"',
            'sed -e "s|<FITS_PATH>|${FITS_PATH}|g" -e "s|<OUTPUT_DIR>|${OUTPUT_DIR}|g" "$CONFIG" > "$RUN_CONFIG"',
            '"$SOFIA_BIN" "$RUN_CONFIG"',
            "",
        ])

    RUN_COMMANDS.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comandos preparados en: {RUN_COMMANDS}")
    print("No se ha ejecutado SoFiA.")


if __name__ == "__main__":
    main()

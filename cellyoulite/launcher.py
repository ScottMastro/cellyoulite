from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn

from cellyoulite.__version__ import __version__


def _open_browser(url: str, delay: float = 1.0) -> None:
    def _go() -> None:
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_go, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(prog="cellyoulite")
    parser.add_argument("command", nargs="?", default="serve",
                        choices=["serve", "version", "simulate", "analyze"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--out", default="./sim-data", help="output folder (simulate)")
    parser.add_argument("--folder", default=None, help="input folder (analyze)")
    args = parser.parse_args()

    if args.command == "version":
        print(__version__)
        return
    if args.command == "simulate":
        from cellyoulite.simulator import generate_dataset
        path = generate_dataset(args.out)
        print(f"wrote synthetic dataset to {path}")
        return
    if args.command == "analyze":
        from cellyoulite.pipeline import analyze_folder
        if not args.folder:
            raise SystemExit("--folder required for analyze")
        df = analyze_folder(args.folder)
        print(df.head().to_string())
        print(f"\n{len(df)} organoids passed QC")
        return

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        _open_browser(url)
    print(f"CellYouLite v{__version__} — serving at {url}")
    uvicorn.run("cellyoulite.server.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

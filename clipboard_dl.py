from pathlib import Path
import argparse
import json
import pyperclip
import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downloads files found in a HTTP archive (HAR) in clipboard, obtained from "
            "Firefox's network tool or your browser's equivalent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("out_dir", type=Path, help=\
        "Path to the output directory.")
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    try:
        j = json.loads(pyperclip.paste())
    except:
        j = None
        print("Couldn't decode clipboard data.")

    try:
        urls = [entry['request']['url'] for entry in j['log']['entries']]
    except:
        urls = []
    print(f"Collected {len(urls)} urls.")

    for url_idx, url in enumerate(urls):
        out_path = args.out_dir / Path(url).name  # Probably not the best way to do this
        if out_path.exists():
            print(f"Skipping {url}, file already exists.")
        else:
            print(f"Downloading {url}")
            req = requests.get(url, allow_redirects=True)
            with open(out_path, 'wb') as fout:
                fout.write(req.content)
    print("Done.")

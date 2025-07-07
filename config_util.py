import yaml


def load_config(path: str) -> dict:
    """YAML ファイルを読み込んで dict を返す"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}

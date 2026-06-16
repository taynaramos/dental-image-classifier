from pathlib import Path


class DataLoader:
    def load(self, path: Path) -> bytes:
        with open(path, "rb") as f:
            return f.read()

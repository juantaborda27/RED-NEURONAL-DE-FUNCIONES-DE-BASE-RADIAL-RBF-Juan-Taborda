# src/main.py
from app import RBF_APP 
from pathlib import Path
import os

DATASETS_FOLDER = Path("datasets")

def main():
    if not DATASETS_FOLDER.exists():
        try:
            os.makedirs(DATASETS_FOLDER)
        except Exception:
            pass
    app = RBF_APP()
    app.mainloop()

if __name__ == "__main__":
    main()

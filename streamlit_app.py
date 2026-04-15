# Compatibility wrapper: keeps the historical entry point while the real UI
# implementation now lives in ui/app.py.
from ui.app import main


if __name__ == "__main__":
    main()

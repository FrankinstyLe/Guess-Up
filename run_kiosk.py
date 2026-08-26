"""Launch the fair kiosk from the repo root.

    python run_kiosk.py
    python run_kiosk.py --no-fullscreen
    python run_kiosk.py --selftest

This exists so the kiosk does not care what directory you are in. Running
`python -m face_match_kiosk.main` also works, but only from inside src/, and at
a fair nobody should have to remember that.
"""

# System imports
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Our imports
from face_match_kiosk.main import main


if __name__ == '__main__':
    sys.exit(main())

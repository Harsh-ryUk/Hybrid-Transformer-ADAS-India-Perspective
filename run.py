import sys
import os

# Ensure the current directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import main

if __name__ == "__main__":
    try:
        # sys.argv is automatically available to argparse in main()
        # No modification needed if we call main() directly, EXCEPT
        # if main() was defined to take explicit args. 
        # Checking src/main.py: it uses parser.parse_args(), so it reads sys.argv directly.
        # This wrapper is correct.
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # input("Press Enter to close...") # Removing input block for headless automation

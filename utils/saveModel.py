from config_locale import BASE
from pathlib import Path

# Create a path object
original_file = Path(f'{BASE}\checkpoint_2500')


# Change the file name completely but keep the folder
new_name = original_file.with_name('best_agent_to_date')

if original_file.exists():
    original_file.rename(new_name)
    print(f"File successfully renamed to: {new_name}")
else:
    print("The original checkpoint file does not exist!")

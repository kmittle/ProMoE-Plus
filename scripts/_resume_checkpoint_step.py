"""Select the highest saved checkpoint step eligible for a training phase."""

import re
import sys
from pathlib import Path


def select_checkpoint_step(directory: Path, minimum: int, target: int) -> int:
    steps = []
    for path in directory.glob('ckpt_step_*.pth'):
        match = re.fullmatch(r'ckpt_step_([1-9][0-9]*)\.pth', path.name)
        if match:
            step = int(match[1])
            if minimum <= step < target and path.is_file() and path.stat().st_size > 0:
                steps.append(step)
    if not steps:
        raise ValueError(f'No nonempty checkpoint in {directory} for {minimum} <= step < {target}')
    return max(steps)


if __name__ == '__main__':
    checkpoint_dir, minimum, target = sys.argv[1:]
    try:
        print(select_checkpoint_step(Path(checkpoint_dir), int(minimum), int(target)))
    except (OSError, ValueError) as exc:
        sys.exit(f'ERROR: {exc}')

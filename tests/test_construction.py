import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


class ConstructionTests(unittest.TestCase):
    def test_checkpoint_to_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'
            base.mkdir()
            original = torch.tensor([1.0, 2.0])
            direction = torch.tensor([0.6, -0.8])
            save_file({'weight': original}, base / 'model.safetensors')
            (base / 'config.json').write_text(json.dumps({'model_type': 'test'}))
            checkpoints = []
            for step in [10, 25, 50]:
                checkpoint = root / f'checkpoint-{step}'
                checkpoint.mkdir()
                save_file({'weight': original + (2 * step + 3) * direction}, checkpoint / 'model.safetensors')
                checkpoints.append(str(checkpoint))
            for mode, coefficient in [('rank1', 503), ('projection', 103), ('endpoint', 515)]:
                output = root / mode
                command = [sys.executable, '-m', 'scare.extrapolate', '--base', str(base), '--checkpoints', *checkpoints, '--steps', '10', '25', '50', '--target', '250', '--mode', mode, '--output', str(output)]
                subprocess.run(command, check=True, capture_output=True, text=True)
                actual = load_file(output / 'model-00001.safetensors')['weight'].numpy()
                np.testing.assert_allclose(actual, (original + coefficient * direction).numpy(), rtol=1e-5)
                metadata = json.loads((output / 'model.safetensors.index.json').read_text())
                self.assertEqual(metadata['metadata']['total_size'], 8)
                self.assertEqual(metadata['weight_map']['weight'], 'model-00001.safetensors')
                self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()

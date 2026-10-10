import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from research.forward_evaluation import forward


class ForwarderTests(unittest.TestCase):
    def test_target_is_pinned_and_runs_in_process(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'run_evaluation_pipeline.py'
            target.write_text("import os\nexecuted_pid=os.getpid()\n")
            record=root/'evaluation_forwarding.json'
            record.write_text(json.dumps({'schema':'cue-study-evaluation-forwarder-v1',
                'target_pipeline':str(target),'target_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
            with contextlib.redirect_stdout(io.StringIO()):result=forward(record)
            import os
            self.assertEqual(result['executed_pid'],os.getpid())
            target.write_text('raise RuntimeError("must not run")')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):forward(record)


if __name__=='__main__':unittest.main()

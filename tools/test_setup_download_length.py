"""Published size drives resume when HEAD and the range probe have no length."""
import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from tools.test_setup_download import DATA, Server, small_pieces, setup


class PublishedDownloadLength(unittest.TestCase):
    def run_download(self, metadata, persistent=False):
        def faults(path, start, end, nth):
            if (start, end) == (0, 0):
                return 'status', 405
            if start == 0 and nth == 1:
                return 'cut', 262144
            if persistent:
                return 'cut', 0

        with Server(length=False, faults=faults, delay=0) as server, tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / 'model.gguf'
            published = (len(DATA), hashlib.sha256(DATA).hexdigest())
            with contextlib.ExitStack() as stack:
                stack.enter_context(small_pieces(DL_MIN_PARALLEL=len(DATA) + 1))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(mock.patch.object(setup, 'file_pin', return_value=None if metadata else published))
                stack.enter_context(mock.patch.object(setup, 'model_source', return_value='modelscope' if metadata else 'huggingface'))
                stack.enter_context(mock.patch.object(setup, 'ms_file', return_value=('repo', 'model.gguf')))
                stack.enter_context(mock.patch.object(setup, 'ms_url', return_value=server.url))
                stack.enter_context(mock.patch.object(setup, 'reachable', return_value=True))
                meta = stack.enter_context(mock.patch.object(setup, 'ms_meta', return_value=published))
                if persistent:
                    with self.assertRaises(SystemExit):
                        setup.download(server.url, dst)
                    self.assertFalse(dst.exists())
                    self.assertFalse(dst.with_name(dst.name + '.done').exists())
                    self.assertEqual(dst.with_name(dst.name + '.part').read_bytes(), DATA[:262144])
                else:
                    setup.download(server.url, dst)
                    self.assertEqual(dst.read_bytes(), DATA)
                    self.assertTrue(dst.with_name(dst.name + '.done').exists())
                self.assertIn((262144, len(DATA) - 1), server.gets)
                self.assertEqual(meta.call_count, 1 if metadata else 0)

    def test_pin_size_recovers_clean_early_eof(self):
        self.run_download(metadata=False)

    def test_modelscope_metadata_size_recovers_clean_early_eof(self):
        self.run_download(metadata=True)

    def test_exhausted_transfer_keeps_resumable_partial(self):
        self.run_download(metadata=False, persistent=True)


if __name__ == '__main__':
    unittest.main()

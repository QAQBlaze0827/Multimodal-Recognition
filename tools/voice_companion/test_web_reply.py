import io
import json
import queue
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import numpy as np

import transcribe_mic as stt
from web_reply import NoRedirect, WebReplyPublisher, validate_web_url


class WebReplyTests(unittest.TestCase):
    def test_loopback_only(self):
        for url in ['http://127.0.0.1:8000', 'http://localhost:8001/', 'http://[::1]:8000']:
            self.assertTrue(validate_web_url(url))
        for url in ['https://example.com', 'http://192.168.1.2:8000', 'http://localhost',
                    'http://localhost:8000/path', 'http://user@localhost:8000',
                    'http://localhost:8000?x=1', 'http://localhost:8000#fragment']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_web_url(url)

    def test_payload_is_text_only_and_sequence_increases(self):
        sender = WebReplyPublisher('http://127.0.0.1:8000')
        response = Mock(status=200)
        response.read.return_value = b'{"accepted":true}'
        manager = Mock(__enter__=Mock(return_value=response), __exit__=Mock(return_value=False))
        sender.opener = Mock()
        sender.opener.open.return_value = manager
        sender._send(message=('我想休息', 'rest', '先休息一下吧'))
        request = sender.opener.open.call_args.args[0]
        data = json.loads(request.data)
        self.assertEqual(set(data), {'session_id', 'sequence', 'state', 'text', 'intent', 'reply'})
        self.assertEqual(data['sequence'], 1)
        self.assertEqual(data['text'], '我想休息')
        self.assertEqual(sender.opener.open.call_args.kwargs['timeout'], 1)
        sender._send(state='stopped')
        data = json.loads(sender.opener.open.call_args.args[0].data)
        self.assertEqual(data['sequence'], 2)
        self.assertNotIn('text', data)

    def test_network_failure_is_not_a_transcription_failure(self):
        sender = WebReplyPublisher('http://127.0.0.1:8000')
        sender.opener = Mock()
        sender.opener.open.side_effect = TimeoutError('unreachable')
        with redirect_stdout(io.StringIO()) as output:
            sender._send(message=('hello', 'clarify', 'reply'))
        self.assertFalse(sender.online)
        self.assertIn('will not be retried', output.getvalue())

    def test_bounded_queue_and_close_reject_late_results(self):
        sender = WebReplyPublisher('http://127.0.0.1:8000')
        with redirect_stdout(io.StringIO()):
            for _ in range(10):
                sender.publish('text', 'company', 'reply')
        self.assertEqual(sender.pending.qsize(), 3)
        sender.close()
        sender.publish('late', 'company', 'late')
        self.assertTrue(sender.pending.empty())

    def test_worker_delivers_reply_and_stop_without_microphone(self):
        sender = WebReplyPublisher('http://127.0.0.1:8000')
        delivered = threading.Event()
        calls = []

        def send(state='listening', message=None):
            calls.append((state, message))
            if message:
                delivered.set()

        with patch.object(sender, '_send', side_effect=send):
            sender.start()
            sender.publish('text', 'company', 'reply')
            self.assertTrue(delivered.wait(3))
            sender.close()
        self.assertEqual(calls[-1][0], 'stopped')
        self.assertFalse(sender.thread.is_alive())

    def test_expired_reply_is_not_replayed(self):
        sender = WebReplyPublisher('http://127.0.0.1:8000')
        sender.pending.put((time.monotonic() - 10, 'old', 'company', 'old'))
        calls = []

        def send(state='listening', message=None):
            calls.append(message)
            if len(calls) >= 2:
                sender.stop.set()

        with patch.object(sender, '_send', side_effect=send), redirect_stdout(io.StringIO()):
            sender.start()
            sender.thread.join(5)
            sender.close()
        self.assertTrue(all(message is None for message in calls))

    def test_transcription_preserves_text_reply_and_no_web_by_default(self):
        phrases, errors, stop = queue.Queue(), queue.SimpleQueue(), threading.Event()
        phrases.put((1, np.zeros(16000, dtype=np.float32), time.perf_counter()))
        publisher = Mock()
        publisher.publish.side_effect = lambda *args: stop.set()
        with patch.object(stt, 'transcribe', return_value=('我想玩', .5)), redirect_stdout(io.StringIO()):
            stt.transcription_worker(Mock(), phrases, stop, errors, publisher)
        publisher.publish.assert_called_once_with('我想玩', 'play', '好呀！我們一起玩！')
        self.assertTrue(errors.empty())

    def test_no_redirects(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com'))

    def test_stream_failure_does_not_start_sender(self):
        sd, sender = Mock(), Mock()
        sd.InputStream.side_effect = RuntimeError('no microphone')
        with self.assertRaises(RuntimeError):
            stt.run(sd, Mock(), None, .002, sender)
        sender.start.assert_not_called()
        sender.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

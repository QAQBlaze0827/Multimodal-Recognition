"""Rule and terminal integration tests; no microphone, model, or audio files."""

import contextlib
import io
import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dog_reply import choose_reply
import transcribe_mic as stt


class ReplyTests(unittest.TestCase):
    def check_cases(self, intent, sentences):
        for text in sentences:
            with self.subTest(text=text):
                result = choose_reply(text)
                self.assertEqual(result[0], intent)
                self.assertTrue(result[1])

    def test_company(self):
        self.check_cases("company", ["陪我一下", "你好，可以陪我一下嗎？", "我想找人陪", "陪陪我"])

    def test_company_reported_live_phrases(self):
        self.check_cases("company", ["你可以陪我一下嗎?", "我希望你可以陪我", "我想要陪伴"])

    def test_company_request_word_order(self):
        self.check_cases("company", ["你能不能陪我一下", "能不能陪我一下", "我希望你陪我", "我想要你陪我", "我需要你的陪伴", "我現在想要陪伴", "你可以陪我一下吗？"])

    def test_company_new_negative_forms(self):
        self.check_cases("quiet", ["不要陪我", "我不想要陪伴", "我不想陪伴", "我不需要陪伴", "我不需要你的陪伴", "我不希望你陪我", "我不要你的陪伴"])
        self.assertEqual(choose_reply("我不想要陪伴")[1], "好，我先不打擾你。")
        self.assertEqual(choose_reply("不要陪我，請先安靜")[1], "好，我先不打擾你。")

    def test_company_uncertain_or_reported_is_not_a_request(self):
        self.check_cases("clarify", ["我不知道要不要你陪我", "我不知道要不要陪伴", "我不希望你可以陪我", "我不是不想要陪伴", "他希望你可以陪我", "我說的是你可以陪我一下嗎", "你可以不要陪我嗎", "你可以陪我一下嗎但我又不想", "我想要陪伴嗎"])

    def test_company_conflicts_and_play_not_swallowed(self):
        self.check_cases("clarify", ["我想要陪伴，我不想要陪伴", "我希望你可以陪我，請先安靜", "我希望你可以陪我，我想玩"])
        self.check_cases("play", ["陪我玩", "可以陪我玩一下"])

    def test_worker_prints_company_for_reported_phrase(self):
        self.assertIn("Dog [company]: 我在這裡，陪你待一下。", self.worker_output("我希望你可以陪我"))

    def test_feeling_and_company_without_punctuation(self):
        self.check_cases("company", [
            "我好難過可不可以陪我一下",
            "我很難過所以我希望你陪我",
            "今天過得不太好我希望你陪我",
            "我現在心情低落你可以陪我一下嗎",
        ])

    def test_feeling_and_company_with_punctuation(self):
        self.check_cases("company", [
            "我好難過，可不可以陪我一下？",
            "今天過得不太好，我希望你陪我",
            "我今天很孤單，所以我想要陪伴",
            "我心情不好，但是我想要你陪我",
            "你好，我很沮喪，可不可以陪我一下",
        ])

    def test_feeling_does_not_override_refusal(self):
        for text in (
            "我好難過，但我不想要你陪我",
            "我好難過但我不想你陪我",
            "今天過得不好，所以我不需要陪伴",
            "我很孤單，可是我不要你的陪伴",
        ):
            with self.subTest(text=text):
                self.assertEqual(choose_reply(text), ("quiet", "好，我先不打擾你。"))

    def test_feeling_only_asks_without_assuming_company(self):
        for text in ("我好難過", "我心情不太好", "今天過得不太好", "我最近很低落"):
            with self.subTest(text=text):
                self.assertEqual(choose_reply(text), ("clarify", "你希望我陪你待一下嗎？"))

    def test_uncertain_companionship_asks(self):
        for text in ("我不知道要不要找人陪", "我不確定想不想你陪我", "我好難過但我不知道要不要找人陪"):
            with self.subTest(text=text):
                self.assertEqual(choose_reply(text), ("clarify", "你希望我陪你待一下嗎？"))

    def test_feeling_context_refuses_unsupported_meanings(self):
        self.check_cases("clarify", [
            "他好難過可不可以陪我一下",
            "朋友說我好難過可不可以陪我一下",
            "我說的是『我好難過可不可以陪我一下』",
            "我不是很難過可不可以陪我一下",
            "我好難過但我不是不想要陪伴",
            "我好難過但我不希望你可以陪我",
            "我好難過嗎",
            "我好難過可能只是累了",
        ])

    def test_feeling_context_conflicts_and_other_intents_stay_conservative(self):
        self.check_cases("clarify", [
            "我好難過，我希望你陪我，但我不想要陪伴",
            "我好難過可不可以陪我一下但不要陪我",
            "我好難過，我想要陪伴，我想玩",
            "我好難過，我想玩",
            "我好難過，我希望你陪我，請先安靜",
        ])

    def test_feeling_context_simplified_and_spaces(self):
        self.check_cases("company", ["我好难过可不可以陪我一下", "我 好 難 過 ， 可 不 可 以 陪 我 一 下 ？", "今天过得不太好，我希望你可以陪我"])

    def test_worker_prints_feeling_request_and_keeps_transcript(self):
        text = "我好難過可不可以陪我一下"
        output = self.worker_output(text)
        self.assertIn(text, output)
        self.assertIn("Dog [company]: 我在這裡，陪你待一下。", output)
        self.assertIn("Dog [clarify]: 你希望我陪你待一下嗎？", self.worker_output("我好難過"))

    def test_play(self):
        self.check_cases("play", ["我想玩", "我現在想玩", "一起玩吧！", "陪我玩一下"])

    def test_rest(self):
        self.check_cases("rest", ["我今天有點累想休息一下", "我今天有點累，想休息一下。", "想休息一下", "我想要休息"])

    def test_quiet(self):
        self.check_cases("quiet", ["我現在不想玩，請先安靜", "現在不想玩,請先安靜", "先安靜一下，我現在不想玩", "不要說話", "請你先安靜一下"])

    def test_negative_only_never_invites_play(self):
        self.check_cases("quiet", ["不想玩", "我不要玩", "不要陪我玩", "我不想你陪我玩", "我不用你陪我", "別陪我"])

    def test_decline_play_but_request_rest(self):
        self.check_cases("rest", ["我不想玩，想休息一下"])

    def test_conflicting_requests(self):
        self.check_cases("clarify", ["我想玩，請先安靜", "想休息一下，一起玩", "我想玩，我不想玩", "不要陪我，陪我一下"])

    def test_unknown_or_uncertain(self):
        self.check_cases("clarify", ["今天在學校遇到一件事情", "我不知道想不想玩", "我不想不玩", "我不想休息", "我不想安靜", "不要安靜，請安靜", "我想玩不想休息", "我覺得不好玩", "想玩嗎"])

    def test_no_substring_matches_or_reported_commands(self):
        self.check_cases("clarify", ["我不想玩具壞掉", "老師說我想玩", "他說請先安靜", "我說的是『不想玩』", "不要叫我休息", "別人想玩"])

    def test_simplified_and_whitespace(self):
        self.check_cases("quiet", ["我现在不想玩，请先安静", "我 現 在 不 想 玩 ， 請 先 安 靜 。"])
        self.check_cases("company", ["可以陪我一会儿吗？"])

    def test_empty_has_no_reply(self):
        for text in ("", "   ", "...", "，。！？"):
            with self.subTest(text=text):
                self.assertIsNone(choose_reply(text))

    def test_rest_request_survives_unclear_adjective(self):
        self.check_cases("rest", ["今天有點內想休息一下"])
        self.check_cases("clarify", ["你先安靜一下，我需要不想玩", "我有點不想休息", "今天有點內"])

    def test_replies_are_stateless(self):
        self.assertEqual(choose_reply("請先安靜")[0], "quiet")
        self.assertEqual(choose_reply("我想玩")[0], "play")

    def worker_output(self, text):
        phrases, errors, stop = queue.Queue(), queue.SimpleQueue(), threading.Event()
        phrases.put((1, np.zeros(stt.BLOCK_SIZE, dtype=np.float32), time.perf_counter()))

        def result(*args):
            stop.set()
            return text, 0.1

        output = io.StringIO()
        with patch.object(stt, "transcribe", side_effect=result), contextlib.redirect_stdout(output):
            stt.transcription_worker(Mock(), phrases, stop, errors)
        self.assertTrue(errors.empty())
        return output.getvalue()

    def test_worker_prints_transcript_and_reply(self):
        output = self.worker_output("我現在不想玩，請先安靜")
        self.assertIn("我現在不想玩，請先安靜", output)
        self.assertIn("Dog [quiet]: 好，我安靜陪著你。", output)
        self.assertIn("STT=0.10s", output)

    def test_worker_empty_does_not_reply(self):
        output = self.worker_output("")
        self.assertIn("No speech recognized", output)
        self.assertNotIn("Dog [", output)


if __name__ == "__main__":
    unittest.main()

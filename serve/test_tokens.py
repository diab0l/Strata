"""serve/test_tokens.py - the vocabulary's own endpoints: POST /tokenize, POST /detokenize, GET /tokenizer.

They answer from the tokenizer alone: no chat template, no roles, no model loaded, no GPU touched.  That is what
they are for.  A protocol whose turn boundaries are token ids has to read this model's ids off this model - the
same ids sit at different numbers in different families (one family's turn markers are 248045 and 248046, another's
130072 and 130073, a third has no end-of-text marker at all) - and reading them is worth doing before a model is
ever loaded, which is the only time a machine short on VRAM has to offer.

    python -m unittest serve.test_tokens -v
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import IM_END, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")

IM_SPECIAL = 256 + ByteTokenizer.SPECIALS.index(IM_END)      # the byte stand-in's ids for its specials


class Bounded(ByteTokenizer):
    """A stand-in tokenizer that does declare its vocabulary, so the id bounds can be tested."""
    tokens = ["a", "b"]


# ------------------------------------------------------------------------------------------------ the service
class Tokens(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "answer"), self.tok, TEMPLATE)

    def test_text_comes_back_as_ids(self):
        self.assertEqual(self.svc.tokenize({"content": "hello"})["tokens"], list(b"hello"))

    def test_a_turn_marker_is_one_id_only_when_specials_are_parsed(self):
        # llama.cpp's rule, measured by strata_tokenizer rather than copied: a CONTROL marker is a marker only
        # with parse_special.  Without it the boundary arrives as the bytes that spell it, which is a prompt with
        # a boundary written as a word.
        self.assertGreater(len(self.svc.tokenize({"content": IM_END, "parse_special": False})["tokens"]), 1)
        self.assertEqual(self.svc.tokenize({"content": IM_END, "parse_special": True})["tokens"], [IM_SPECIAL])

    def test_parse_special_is_the_default_so_a_plain_call_still_writes_boundaries_as_ids(self):
        self.assertEqual(self.svc.tokenize({"content": IM_END})["tokens"], [IM_SPECIAL])

    def test_an_id_reads_back_as_the_marker_it_is(self):
        self.assertEqual(self.svc.detokenize({"tokens": [IM_SPECIAL]})["content"], IM_END)

    def test_an_id_outside_the_vocabulary_is_refused(self):
        tok = Bounded()
        svc = Service(MockEngine(tok, "answer"), tok, TEMPLATE)
        with self.assertRaises(ValueError):
            svc.detokenize({"tokens": [7]})
        self.assertEqual(svc.detokenize({"tokens": [1]})["content"], "\u0001")   # ids as bytes, as the stand-in decodes

    def test_add_special_uses_the_vocabularys_own_bos_and_eos(self):
        # The byte stand-in declares none, so asking for them is a refusal, not a prompt with nothing added.
        with self.assertRaises(ValueError):
            self.svc.tokenize({"content": "hi", "add_special": True})

    def test_a_bad_request_is_a_value_error_the_handler_answers_as_400(self):
        for bad in ({}, {"content": 5}, {"content": "x", "parse_special": "yes"}, {"content": "x", "add_special": 1}):
            with self.assertRaises(ValueError):
                self.svc.tokenize(bad)
        for bad in ({}, {"tokens": []}, {"tokens": "1"}, {"tokens": [1, True]}, {"tokens": [1], "add_special": "x"}):
            with self.assertRaises(ValueError):
                self.svc.detokenize(bad)

    def test_the_vocabulary_lists_its_markers_with_their_ids_and_kind(self):
        info = self.svc.tokenizer_info()
        listed = {m["marker"]: m for m in info["markers"]}
        self.assertEqual(listed[IM_END], {"id": IM_SPECIAL, "marker": IM_END, "kind": "control"})
        self.assertEqual(info["model"], "qwen3.8-flash-next")
        self.assertNotIn("vocab", info)                      # the byte stand-in has no vocabulary to size


# ------------------------------------------------------------------------------------------------ over HTTP
class OverHttp(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "answer"), self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def call(self, method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request(method, path, body=None if body is None else json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
                r = c.getresponse()
                return r.status, json.loads(r.read().decode())
        finally:
            c.close()

    def test_an_id_goes_out_and_its_marker_comes_back(self):
        code, r = self.call("POST", "/tokenize", {"content": IM_END, "parse_special": True})
        self.assertEqual(code, 200)
        self.assertEqual(r["tokens"], [IM_SPECIAL])
        code, back = self.call("POST", "/detokenize", {"tokens": r["tokens"]})
        self.assertEqual((code, back["content"]), (200, IM_END))

    def test_the_endpoints_are_not_404_anymore(self):
        # llama.cpp has these; a client that speaks ids probes them and needs the same answers here.
        for path in ("/tokenize", "/detokenize"):
            self.assertNotEqual(self.call("POST", path, {})[0], 404)

    def test_the_vocabulary_answers_before_anything_was_generated(self):
        code, info = self.call("GET", "/tokenizer")
        self.assertEqual(code, 200)
        self.assertTrue(any(m["marker"] == IM_END for m in info["markers"]))
        self.assertFalse(self.svc.status.get("busy"))        # no request ran to answer this one

    def test_a_bad_request_is_a_400_that_says_what_was_wrong(self):
        code, r = self.call("POST", "/detokenize", {"tokens": []})
        self.assertEqual((code, r["error"]["type"]), (400, "invalid_request_error"))
        self.assertIn("tokens", r["error"]["message"])


if __name__ == "__main__":
    unittest.main()

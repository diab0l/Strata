"""serve/test_completion.py - POST /completion: token soup in, token soup out, against the mock engine (no GPU, no pack).

The route exists for clients whose protocol is token ids rather than messages: a turn protocol that writes a turn
boundary as an id and ends a turn by reading one needs a server that hands the ids over - the model's own end-of-turn
id included - and that stops paying attention when the client stops reading.

    python -m unittest serve.test_completion -v
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import IM_END, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")

IM_SPECIAL = 256 + ByteTokenizer.SPECIALS.index(IM_END)
BODY = list(b"hello there")


class Declared(ByteTokenizer):
    """A stand-in tokenizer that does declare its vocabulary, so the id bounds can be tested."""
    tokens = [chr(i) for i in range(128)]


def raw(svc, ids, **req):
    """Run one raw completion to the end: -> ([ids], the done figures)."""
    cancel = threading.Event()
    out, done = [], None
    for kind, x in svc.complete_raw(ids, req.get("n_predict"), req, cancel):
        if kind == "token":
            out.append(x)
        elif kind == "done":
            done = x
    return out, done


# ------------------------------------------------------------------------------------------------ the service
class Completion(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "hello there", max_context=1024)
        self.svc = Service(self.engine, self.tok, TEMPLATE)

    def test_the_ids_that_come_back_are_the_ones_the_model_produced(self):
        out, done = raw(self.svc, list(b"prompt"))
        self.assertEqual(out, BODY + [IM_SPECIAL])
        self.assertEqual(done["finish"], "eos")
        self.assertEqual(done["generated"], len(out))

    def test_the_end_of_turn_id_arrives_as_an_id_rather_than_as_an_absence(self):
        # A client that ends a turn by reading this id cannot do it if the server eats the id and only reports
        # that something happened.
        out, _ = raw(self.svc, list(b"prompt"))
        self.assertIn(out[-1], self.svc.stop_ids)

    def test_n_predict_is_a_limit_and_says_so(self):
        out, done = raw(self.svc, list(b"prompt"), n_predict=3)
        self.assertEqual(out, BODY[:3])
        self.assertEqual(done["finish"], "length")

    def test_no_n_predict_means_the_room_left_in_the_context(self):
        out, done = raw(self.svc, list(b"prompt"))
        self.assertEqual(done["finish"], "eos")                        # the script ends long before the room does
        out2, done2 = raw(self.svc, list(b"prompt"), n_predict=-1)    # llama.cpp's "no limit": the room
        self.assertEqual((out, done["generated"]), (out2, done2["generated"]))

    def test_a_prompt_is_ids_or_text_or_the_two_mixed(self):
        self.assertEqual(self.svc.raw_prompt({"prompt": "hi"}), list(b"hi"))
        self.assertEqual(self.svc.raw_prompt({"prompt": [65, "b"]}), [65, 98])
        self.assertEqual(self.svc.raw_prompt({"prompt": [IM_SPECIAL]}), [IM_SPECIAL])
        self.assertEqual(self.svc.raw_prompt({"prompt": IM_END}), [IM_SPECIAL])   # a marker written as text is one

    def test_a_prompt_of_the_wrong_shape_is_refused(self):
        for bad in ({}, {"prompt": []}, {"prompt": None}, {"prompt": 5}, {"prompt": [None]}, {"prompt": [1.5]}):
            with self.assertRaises(ValueError):
                self.svc.raw_prompt(bad)

    def test_an_id_that_is_not_in_this_vocabulary_is_refused(self):
        # A wrong id is a wrong prompt: the engine would answer with whatever the model makes of it.  A
        # stand-in tokenizer that declares nothing has no bound to check against, so this one declares 128.
        tok = Declared()
        svc = Service(MockEngine(tok, "hi"), tok, TEMPLATE)
        with self.assertRaises(ValueError) as e:
            svc.raw_prompt({"prompt": [999999]})
        self.assertIn("not an id in this vocabulary", str(e.exception))
        self.assertEqual(svc.raw_prompt({"prompt": [65, 66]}), [65, 66])

    def test_a_prompt_that_leaves_no_room_to_answer_is_refused(self):
        with self.assertRaises(ValueError) as e:
            raw(self.svc, list(range(1024)))
        self.assertIn("leaves no room to answer", str(e.exception))

    def test_too_much_asked_for_is_refused_or_clamped_by_fit_max_tokens(self):
        with self.assertRaises(ValueError) as e:
            raw(self.svc, list(b"prompt"), n_predict=4096)
        self.assertIn("n_predict", str(e.exception))
        self.assertIn("Send a smaller n_predict", str(e.exception))
        self.engine = MockEngine(self.tok, "".join(chr(97 + i % 26) for i in range(2000)),
                                     max_context=1024)   # more answer than the context holds
        self.svc.engine = self.engine
        self.svc.fit_max_tokens = True                     # the config's "fit_max_tokens": true (#545)
        out, done = raw(self.svc, list(b"prompt"), n_predict=4096)
        self.assertEqual(done["finish"], "length")                   # a shorter answer beats a 400
        self.assertEqual(done["generated"], 1024 - 8 - 6)

    def test_the_reply_carries_no_text_for_a_control_token(self):
        # llama.cpp's decode adds nothing for a CONTROL token, and a client reading a body out of the text
        # channel must not get the boundary's spelling in it.
        tok = ByteTokenizer()
        engine = MockEngine(tok, "a" + IM_END + "b", max_context=1024)   # the script writes the boundary itself
        svc = Service(engine, tok, TEMPLATE)
        out, done = raw(svc, list(b"prompt"))
        self.assertEqual(out[-1], IM_SPECIAL)                # the turn ended there: "b" was never reached
        self.assertEqual(done["finish"], "eos")
        self.assertEqual([t for t in out if t not in svc._control_ids()], [97])

    def test_a_client_that_stops_reading_ends_as_a_disconnect(self):
        gen = self.svc.complete_raw(list(b"prompt"), None, {}, threading.Event())
        for _ in range(2):
            next(gen)
        gen.close()                                   # the protocol's decision: this turn is over
        self.assertEqual(self.svc.history[-1]["finish"], "disconnect")
        self.assertFalse(self.svc.status.get("busy"))

    def test_a_cancelled_request_says_cancel(self):
        cancel = threading.Event()
        out = []
        for kind, x in self.svc.complete_raw(list(b"prompt"), None, {}, cancel):
            if kind == "token":
                out.append(x)
                cancel.set()
        self.assertEqual(out[:1], BODY[:1])
        self.assertEqual(self.svc.history[-1]["finish"], "cancel")

    def test_every_raw_request_is_counted_like_any_other(self):
        raw(self.svc, list(b"prompt"))
        self.assertEqual(self.svc.totals["requests"], 1)
        self.assertEqual(self.svc.totals["output_tokens"], len(BODY) + 1)
        self.assertEqual(len(self.svc.history), 1)


# ------------------------------------------------------------------------------------------------ over HTTP
class OverHttp(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "hello there", max_context=1024)
        self.svc = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body):
        """(status, parsed body, or the list of data: events when the answer streamed)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", "/completion", body=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
                r = c.getresponse()
                raw_body = r.read().decode()
            if r.getheader("Content-Type") == "text/event-stream":
                return r.status, [json.loads(l[len("data: "):]) for l in raw_body.split("\n\n")
                                  if l.startswith("data: ") and not l.endswith("[DONE]")]
            return r.status, json.loads(raw_body)
        finally:
            c.close()

    def test_ids_go_out_and_ids_come_back(self):
        code, r = self.post({"prompt": [112, 114, 111, 109, 112, 116], "n_predict": -1})
        self.assertEqual(code, 200)
        self.assertEqual(r["content"], "hello there")           # no marker text in the body
        self.assertEqual(r["tokens"], BODY + [IM_SPECIAL])      # and the boundary is in the ids
        self.assertEqual((r["stop_type"], r["truncated"]), ("eos", False))
        self.assertEqual(r["tokens_predicted"], len(BODY) + 1)
        self.assertEqual(r["tokens_evaluated"], 6)

    def test_return_tokens_false_keeps_the_body_and_drops_the_ids(self):
        code, r = self.post({"prompt": "prompt", "return_tokens": False})
        self.assertEqual((code, r["content"]), (200, "hello there"))
        self.assertNotIn("tokens", r)

    def test_a_stream_hands_over_one_id_at_a_time_and_then_the_figures(self):
        code, events = self.post({"prompt": "prompt", "stream": True})
        self.assertEqual(code, 200)
        self.assertEqual([e["tokens"][0] for e in events if e["tokens"]], BODY + [IM_SPECIAL])
        self.assertEqual("".join(e["content"] for e in events), "hello there")
        last = events[-1]
        self.assertEqual((last["tokens"], last["stop"], last["stop_type"]), ([], True, "eos"))
        self.assertEqual(last["tokens_predicted"], len(BODY) + 1)

    def test_a_stream_can_be_read_until_the_boundary_and_no_further(self):
        # What the protocol actually does: read ids until the turn-closing one, then stop reading.  Nothing here
        # told the server where that was.
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", "/completion", body=json.dumps({"prompt": "prompt", "stream": True}).encode(),
                          headers={"Content-Type": "application/json"})
                r = c.getresponse()
                seen = []
                while True:
                    line = r.readline()
                    if not line:
                        break
                    if line.startswith(b"data: "):
                        ev = json.loads(line[6:].decode())
                        ids = ev.get("tokens") or []
                        if ids:
                            seen.append(ids[0])
                        if not ids or ids[0] == IM_SPECIAL:
                            break
        finally:
            c.close()
        self.assertEqual(seen[-1], IM_SPECIAL)
        self.assertNotIn(IM_SPECIAL, seen[:-1])

    def test_stop_strings_are_refused_rather_than_ignored(self):
        code, r = self.post({"prompt": "hi", "stop": ["end"]})
        self.assertEqual(code, 400)
        self.assertIn("does not stop on text", r["error"]["message"])

    def test_a_bad_prompt_is_a_400_before_the_model_is_asked_for_anything(self):
        self.tok = Declared()                   # a vocabulary with a bound to check an id against
        self.svc.tok = self.tok
        self.svc.engine = MockEngine(self.tok, "hello there", max_context=1024)
        code, r = self.post({"prompt": [999999]})
        self.assertEqual((code, r["error"]["type"]), (400, "invalid_request_error"))
        self.assertIn("not an id in this vocabulary", r["error"]["message"])
        code, r = self.post({"prompt": "prompt", "n_predict": "lots"})
        self.assertIn("n_predict", r["error"]["message"])

    def test_a_reply_that_keeps_going_is_still_bounded_by_n_predict(self):
        self.engine = MockEngine(self.tok, "x" * 400, max_context=1024)
        self.svc.engine = self.engine
        code, r = self.post({"prompt": "prompt", "n_predict": 40})
        self.assertEqual((code, r["stop_type"], r["truncated"]), (200, "limit", True))
        self.assertEqual(r["tokens_predicted"], 40)

    def test_the_chat_fields_are_not_read_here(self):
        # No messages, no template, no tools: bringing them changes nothing, which is the point of the route.
        code, r = self.post({"prompt": "prompt", "messages": [{"role": "user", "content": "hi"}],
                             "tools": [{"type": "function", "name": "x"}]})
        self.assertEqual((code, r["content"]), (200, "hello there"))


if __name__ == "__main__":
    unittest.main()

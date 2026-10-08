"""Tests for LM Studio integration (OpenAI-compatible API)."""

import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import ocr_service
from ocr_service import OCRRequest, OCRServiceError


def sse_chunk(text):
    payload = {"choices": [{"delta": {"content": text}}]}
    return "data: " + json.dumps(payload)


def make_sse_response(lines):
    resp = mock.MagicMock()
    resp.iter_lines.return_value = iter(lines)
    resp.raise_for_status.return_value = None
    return resp


class TestListModelsLMStudio(unittest.TestCase):
    URL = "http://localhost:1234/v1"

    def test_fetch_models_from_lmstudio(self):
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen2.5-vl-7b-instruct"},
                {"id": "llava-v1.6-mistral-7b"},
                {"id": "qwen2.5-vl-7b-instruct"},
            ]
        }
        mock_response.raise_for_status.return_value = None

        with mock.patch("requests.get", return_value=mock_response) as mock_get:
            result = ocr_service.list_models_lmstudio(self.URL)

        self.assertEqual(
            result,
            ["llava-v1.6-mistral-7b", "qwen2.5-vl-7b-instruct"],
        )
        mock_get.assert_called_once_with(
            self.URL + "/models",
            timeout=config.MODEL_LIST_TIMEOUT,
        )

    def test_url_without_v1_gets_suffix(self):
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"data": [{"id": "m1"}]}
        mock_response.raise_for_status.return_value = None

        with mock.patch("requests.get", return_value=mock_response) as mock_get:
            result = ocr_service.list_models_lmstudio("http://localhost:1234")

        self.assertEqual(result, ["m1"])
        called_url = mock_get.call_args[0][0]
        self.assertEqual(called_url, "http://localhost:1234/v1/models")

    def test_empty_model_list(self):
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None

        with mock.patch("requests.get", return_value=mock_response):
            result = ocr_service.list_models_lmstudio(self.URL)

        self.assertEqual(result, [])

    def test_connection_error_wrapped(self):
        with mock.patch("requests.get", side_effect=ConnectionError("refused")):
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.list_models_lmstudio(self.URL)

        self.assertIn("refused", str(ctx.exception))

    def test_list_models_dispatches_by_backend(self):
        with mock.patch.object(
            ocr_service, "list_models_lmstudio", return_value=["a"]
        ) as lm:
            result = ocr_service.list_models(self.URL, config.BACKEND_LMSTUDIO)
        lm.assert_called_once_with(self.URL)
        self.assertEqual(result, ["a"])


class TestRecognizeImagesLMStudio(unittest.TestCase):
    MODEL = "qwen2.5-vl-7b-instruct"
    URL = "http://localhost:1234/v1"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def make_image(self, name="test.png"):
        from PIL import Image

        path = self.dir / name
        Image.new("RGB", (100, 100), (255, 0, 0)).save(path)
        return path

    def make_client(self, url=None):
        class LMStudioClient:
            def __init__(self, base_url):
                self.base_url = base_url

        return LMStudioClient(url or self.URL)

    def test_recognize_single_image(self):
        image_path = self.make_image()
        client = self.make_client()
        lines = [sse_chunk("Hello"), sse_chunk(" world"), "data: [DONE]"]

        with mock.patch(
            "requests.post", return_value=make_sse_response(lines)
        ) as mock_post:
            result = ocr_service.recognize_images_lmstudio(
                client, self.MODEL, [image_path], lambda _msg: None
            )

        self.assertEqual(result, ["Hello world"])
        mock_post.assert_called_once()
        called_url = mock_post.call_args[0][0]
        self.assertIn("/chat/completions", called_url)

    def test_recognize_multiple_images(self):
        image_paths = [self.make_image("page_%d.png" % i) for i in range(2)]
        client = self.make_client()

        def post_side_effect(*args, **kwargs):
            idx = len(post_side_effect.calls)
            post_side_effect.calls.append(kwargs)
            return make_sse_response([sse_chunk("Page %d" % idx), "data: [DONE]"])

        post_side_effect.calls = []

        with mock.patch("requests.post", side_effect=post_side_effect):
            result = ocr_service.recognize_images_lmstudio(
                client, self.MODEL, image_paths, lambda _msg: None
            )

        self.assertEqual(result, ["Page 0", "Page 1"])

    def test_empty_response_fails(self):
        image_path = self.make_image()
        client = self.make_client()

        with mock.patch(
            "requests.post", return_value=make_sse_response(["data: [DONE]"])
        ):
            with self.assertRaises(OCRServiceError) as ctx:
                ocr_service.recognize_images_lmstudio(
                    client, self.MODEL, [image_path], lambda _msg: None
                )

        self.assertIn("returned no text", str(ctx.exception))

    def test_request_format(self):
        image_path = self.make_image()
        client = self.make_client()

        with mock.patch(
            "requests.post",
            return_value=make_sse_response([sse_chunk("text"), "data: [DONE]"]),
        ) as mock_post:
            ocr_service.recognize_images_lmstudio(
                client, self.MODEL, [image_path], lambda _msg: None
            )

        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["model"], self.MODEL)
        self.assertTrue(payload["stream"])
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")

        content = payload["messages"][1]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(
            content[1]["image_url"]["url"].startswith("data:image/")
        )

    def test_stream_chunk_and_page_text_events(self):
        image_path = self.make_image()
        client = self.make_client()
        lines = [sse_chunk("A"), sse_chunk("B"), "data: [DONE]"]
        events = []

        with mock.patch(
            "requests.post", return_value=make_sse_response(lines)
        ):
            result = ocr_service.recognize_images_lmstudio(
                client,
                self.MODEL,
                [image_path],
                lambda _msg: None,
                event_callback=lambda k, p: events.append((k, p)),
            )

        self.assertEqual(result, ["AB"])
        chunks = [p for k, p in events if k == "stream_chunk"]
        self.assertEqual(
            chunks, [{"page": 1, "text": "A"}, {"page": 1, "text": "B"}]
        )
        pages = [p for k, p in events if k == "page_text"]
        self.assertEqual(pages, [{"page": 1, "total": 1, "text": "AB"}])

    def test_recognize_images_dispatches_by_backend(self):
        image_path = self.make_image()
        client = self.make_client()
        with mock.patch.object(
            ocr_service, "recognize_images_lmstudio", return_value=["x"]
        ) as lm:
            result = ocr_service.recognize_images(
                client,
                self.MODEL,
                [image_path],
                lambda _msg: None,
                backend=config.BACKEND_LMSTUDIO,
            )
        self.assertEqual(result, ["x"])
        lm.assert_called_once()


class TestProcessOcrLMStudio(unittest.TestCase):
    URL = "http://localhost:1234/v1"
    MODEL = "qwen2.5-vl-7b-instruct"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_process_image_with_lmstudio(self):
        from PIL import Image

        input_path = self.dir / "test.png"
        Image.new("RGB", (100, 100), (255, 0, 0)).save(input_path)

        request = OCRRequest(
            input_path=input_path,
            output_path=ocr_service.build_output_path(input_path),
            ollama_url=self.URL,
            model=self.MODEL,
            dpi=150,
            backend=config.BACKEND_LMSTUDIO,
        )

        lines = [sse_chunk("Hello LM Studio"), "data: [DONE]"]
        events = queue.Queue()
        with mock.patch(
            "requests.post", return_value=make_sse_response(lines)
        ):
            result = ocr_service.process_ocr(request, events)

        self.assertEqual(result, request.output_path)
        self.assertEqual(
            request.output_path.read_text(encoding="utf-8"), "Hello LM Studio"
        )


if __name__ == "__main__":
    unittest.main()

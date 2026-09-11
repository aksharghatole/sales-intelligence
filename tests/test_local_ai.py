import unittest
from types import SimpleNamespace
from unittest.mock import patch

from local_ai import (
    normalize_ollama_url,
    classify_ollama_environment,
    diagnose_ollama,
    generate_ai,
    test_generation,
    build_ollama_request_proxies,
)


class LocalAITestCase(unittest.TestCase):
    def test_normalize_ollama_url_adds_scheme_and_strips_secrets(self):
        self.assertEqual(normalize_ollama_url("127.0.0.1:11434"), "http://127.0.0.1:11434")
        self.assertEqual(normalize_ollama_url("https://user:pass@host:11434/api"), "https://host:11434/api")

    def test_localhost_configuration_is_local(self):
        self.assertEqual(classify_ollama_environment("http://127.0.0.1:11434"), "local")
        self.assertEqual(classify_ollama_environment("http://localhost:11434"), "local")

    def test_private_ip_detection_is_private(self):
        self.assertEqual(classify_ollama_environment("http://192.168.1.22:11434"), "private")
        self.assertEqual(classify_ollama_environment("http://10.0.0.1:11434"), "private")
        self.assertEqual(classify_ollama_environment("http://172.27.94.123:11434"), "private")

    def test_diagnose_ollama_handles_malformed_response(self):
        class BadResponse:
            status_code = 200

            def json(self):
                raise ValueError("bad json")

        with patch("local_ai.requests.get", return_value=BadResponse()):
            diag = diagnose_ollama()
            self.assertEqual(diag["error_type"], "malformed_json")

    def test_diagnose_ollama_handles_model_not_found_response(self):
        class Resp:
            status_code = 200

            def json(self):
                return {"models": [{"name": "other-model:latest"}]}

        with patch("local_ai.requests.get", return_value=Resp()):
            diag = diagnose_ollama()
            self.assertEqual(diag["error_type"], "model_not_found")
            self.assertFalse(diag["model_available"])

    def test_diagnose_ollama_handles_successful_health_response(self):
        class Resp:
            status_code = 200

            def json(self):
                return {"models": [{"name": "llama3.2:1b"}]}

        with patch("local_ai.requests.get", return_value=Resp()):
            diag = diagnose_ollama()
            self.assertEqual(diag["error_type"], "none")
            self.assertTrue(diag["model_available"])

    def test_generate_ai_success_uses_mocked_http_payload(self):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"response": "BWC Ollama connection working."}

        with patch("local_ai.requests.post", return_value=Resp()):
            result = generate_ai("Reply with exactly: BWC Ollama connection working.")
            self.assertEqual(result, "BWC Ollama connection working.")

    def test_build_ollama_request_proxies_returns_none_when_proxy_empty(self):
        with patch("local_ai.settings", SimpleNamespace(ollama_proxy="")):
            self.assertIsNone(build_ollama_request_proxies())

    def test_build_ollama_request_proxies_returns_socks_mapping_when_configured(self):
        with patch("local_ai.settings", SimpleNamespace(ollama_proxy="socks5h://127.0.0.1:1055")):
            self.assertEqual(
                build_ollama_request_proxies(),
                {
                    "http": "socks5h://127.0.0.1:1055",
                    "https": "socks5h://127.0.0.1:1055",
                },
            )

    def test_diagnose_ollama_health_check_uses_proxy_when_configured(self):
        class Resp:
            status_code = 200

            def json(self):
                return {"models": [{"name": "llama3.2:1b"}]}

        fake_settings = SimpleNamespace(
            ai_mode="ollama",
            ai_timeout=60,
            ollama_model="llama3.2:1b",
            ollama_url="http://127.0.0.1:11434",
            ollama_proxy="socks5h://127.0.0.1:1055",
        )
        with patch("local_ai.settings", fake_settings):
            with patch("local_ai.requests.get") as mock_get:
                mock_get.return_value = Resp()
                diag = diagnose_ollama()
                self.assertEqual(diag["error_type"], "none")
                self.assertEqual(mock_get.call_args.kwargs["proxies"], {"http": "socks5h://127.0.0.1:1055", "https": "socks5h://127.0.0.1:1055"})

    def test_test_generation_success_uses_mocked_http_payload(self):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"response": "BWC Ollama connection working."}

        with patch("local_ai.ollama_enabled", return_value=True), patch("local_ai.diagnose_ollama", return_value={"configured": True, "reachable": True, "model_available": True, "ollama_running": True, "environment": "local", "message": "ok", "error_type": "none", "url": "http://127.0.0.1:11434", "model": "llama3.2:1b"}), patch("local_ai.requests.post", return_value=Resp()):
            result = test_generation("Reply with exactly: BWC Ollama connection working.")
            self.assertTrue(result["success"])
            self.assertIn("BWC Ollama connection working.", result["response"])


if __name__ == "__main__":
    unittest.main()

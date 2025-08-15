import unittest
from unittest.mock import patch, MagicMock
import requests
from src.web import WebBrowser

class TestWebBrowser(unittest.TestCase):

    def setUp(self):
        self.browser = WebBrowser()

    @patch('requests.Session.get')
    def test_fetch_page_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>Test</title></head><body><p>Hello</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        content = self.browser.fetch_page("http://example.com")
        self.assertEqual(content, "<html><head><title>Test</title></head><body><p>Hello</p></body></html>")

    @patch('requests.Session.get')
    def test_fetch_page_failure(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Test error")

        content = self.browser.fetch_page("http://example.com")
        self.assertTrue(content.startswith("Error:"))

    def test_get_text_from_html(self):
        html = "<html><head><title>Test</title><style>.p {color: red;}</style></head><body><p>Hello</p><script>alert('world')</script></body></html>"
        soup = self.browser.parse_html(html)
        text = self.browser.get_text_from_html(soup)
        self.assertEqual(text, "Test Hello")

    def test_summarize_text(self):
        long_text = "a" * 1000
        summary = self.browser.summarize_text(long_text, max_length=100)
        self.assertEqual(len(summary), 103)
        self.assertTrue(summary.endswith("..."))

    @patch('requests.Session.get')
    def test_browse_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>Test</title></head><body><p>Hello World</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        summary = self.browser.browse("http://example.com")
        self.assertEqual(summary, "Test Hello World")

if __name__ == '__main__':
    unittest.main()

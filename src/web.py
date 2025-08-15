"""
Web Browsing Module

This module provides the web browsing capabilities for the AI assistant.
It uses requests to fetch web pages and BeautifulSoup to parse them.
"""

import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class WebBrowser:
    """A simple web browser that can fetch and parse web pages."""

    def __init__(self, timeout: int = 10):
        """
        Initialize the WebBrowser.

        Args:
            timeout: Timeout for web requests in seconds.
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        self.timeout = timeout

    def fetch_page(self, url: str) -> str:
        """
        Fetch the content of a web page.

        Args:
            url: The URL of the web page to fetch.

        Returns:
            The content of the web page as a string, or an error message.
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return f"Error: Could not fetch the URL. {e}"

    def parse_html(self, html_content: str) -> BeautifulSoup:
        """
        Parse HTML content using BeautifulSoup.

        Args:
            html_content: The HTML content to parse.

        Returns:
            A BeautifulSoup object.
        """
        return BeautifulSoup(html_content, "lxml")

    def get_text_from_html(self, soup: BeautifulSoup) -> str:
        """
        Extract text from a BeautifulSoup object.

        Args:
            soup: A BeautifulSoup object.

        Returns:
            The extracted text.
        """
        for script in soup(["script", "style"]):
            script.extract()
        return " ".join(soup.get_text(separator=" ").split())

    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """
        Summarize text to a maximum length.

        Args:
            text: The text to summarize.
            max_length: The maximum length of the summary.

        Returns:
            The summarized text.
        """
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def browse(self, url: str) -> str:
        """
        Browse a URL and return a summary of its content.

        Args:
            url: The URL to browse.

        Returns:
            A summary of the web page content, or an error message.
        """
        logger.info(f"Browsing URL: {url}")
        html_content = self.fetch_page(url)
        if html_content.startswith("Error:"):
            return html_content

        soup = self.parse_html(html_content)
        page_text = self.get_text_from_html(soup)
        summary = self.summarize_text(page_text)

        return summary

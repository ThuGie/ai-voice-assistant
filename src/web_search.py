"""
Web Search Engine Module

This module provides the functionality to search the web for information.
"""

import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class WebSearchEngine:
    """
    A class to perform web searches and parse the results.
    """
    def __init__(self, user_agent="AI-Voice-Assistant/1.0"):
        """
        Initialize the web search engine.

        Args:
            user_agent: The User-Agent string to use for web requests.
        """
        self.user_agent = user_agent
        self.headers = {"User-Agent": self.user_agent}

    def search(self, query: str, num_results: int = 3) -> list[dict[str, str]]:
        """
        Perform a web search for the given query.

        Args:
            query: The search query.
            num_results: The number of results to return.

        Returns:
            A list of search results, where each result is a dictionary
            containing the 'title', 'link', and 'snippet' of the result.
        """
        search_url = f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}"
        logger.info(f"Performing web search for: {query}")

        try:
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            for result in soup.find_all("div", class_="result"):
                title_tag = result.find("a", class_="result__a")
                snippet_tag = result.find("a", class_="result__snippet")

                if title_tag and snippet_tag:
                    title = title_tag.get_text(strip=True)
                    link = title_tag["href"]
                    snippet = snippet_tag.get_text(strip=True)

                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })

                if len(results) >= num_results:
                    break

            logger.info(f"Found {len(results)} results for query: {query}")
            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to perform web search: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during web search: {e}")
            return []

if __name__ == "__main__":
    # Example usage
    search_engine = WebSearchEngine()
    test_query = "What is the capital of France?"
    search_results = search_engine.search(test_query)

    if search_results:
        print(f"Search results for: '{test_query}'")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. {result['title']}")
            print(f"   Link: {result['link']}")
            print(f"   Snippet: {result['snippet']}")
            print("-" * 20)
    else:
        print(f"No search results found for: '{test_query}'")

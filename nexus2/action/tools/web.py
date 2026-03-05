"""Web tools: web_search, web_fetch, wikipedia."""

import logging
import re

from ..tool_registry import ToolResult


class WebSearchTool:
    name = "web_search"
    description = (
        "Search the web for current information. "
        "Usage: [TOOL_CALL: web_search | your search query here]"
    )

    def run(self, query: str) -> ToolResult:
        try:
            from ddgs import DDGS
        except ImportError:
            return ToolResult(self.name, "", success=False,
                              error="ddgs not installed. Run: pip install ddgs")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query.strip(), max_results=4))
            if not results:
                return ToolResult(self.name, "No results found.")
            lines = []
            for r in results:
                title = r.get("title", "").strip()
                snippet = r.get("body", "").strip()
                url = r.get("href", "").strip()
                lines.append(f"* {title}\n  {snippet}\n  {url}")
            return ToolResult(self.name, "\n\n".join(lines))
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class WebFetchTool:
    name = "web_fetch"
    description = (
        "Fetch and extract text from a URL. "
        "Usage: [TOOL_CALL: web_fetch | https://example.com]"
    )

    def run(self, url: str) -> ToolResult:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return ToolResult(self.name, "", success=False,
                              error="requests/beautifulsoup4 not installed")
        try:
            url = url.strip()
            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "NEXUS-2 Agent/1.0"
            })
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove script/style
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Truncate
            if len(text) > 4000:
                text = text[:4000] + "\n...[truncated]"
            return ToolResult(self.name, text)
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))


class WikipediaTool:
    name = "wikipedia"
    description = (
        "Get a Wikipedia summary for a topic. "
        "Usage: [TOOL_CALL: wikipedia | topic name]"
    )

    def run(self, topic: str) -> ToolResult:
        try:
            import requests
        except ImportError:
            return ToolResult(self.name, "", success=False,
                              error="requests not installed")
        try:
            topic = topic.strip()
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "NEXUS-2 Agent/1.0"
            })
            if resp.status_code == 404:
                return ToolResult(self.name, f"No Wikipedia article found for '{topic}'.")
            resp.raise_for_status()
            data = resp.json()
            extract = data.get("extract", "No summary available.")
            title = data.get("title", topic)
            return ToolResult(self.name, f"{title}: {extract}")
        except Exception as e:
            return ToolResult(self.name, "", success=False, error=str(e))

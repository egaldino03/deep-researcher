from unittest.mock import patch

from app.agent.tools import search


@patch("app.agent.tools.TavilySearch")
def test_search_tool_mocked(mock_tavily_class):
    # 1. Setup the mock TavilySearch instance
    mock_instance = mock_tavily_class.return_value
    mock_instance.invoke.return_value = [
        {
            "url": "https://example.com/arch",
            "content": "Arch Linux Hyprland configuration guide",
        }
    ]

    # 2. Invoke the tool (note: LangChain tools expect a dictionary input for single-arg strings usually, or string depending on how it's defined)
    result = search.invoke({"query": "Arch Linux Hyprland configuration guide."})

    # 3. Assert TavilySearch was initialized correctly inside your tool
    mock_tavily_class.assert_called_once_with(
        max_result=12, topic="general", search_depth="advanced"
    )

    # 4. Assert the inner tavily client's invoke was called with the query
    mock_instance.invoke.assert_called_once_with(
        {"query": "Arch Linux Hyprland configuration guide."}
    )

    # 5. Assert the result
    assert len(result) == 1
    assert "Hyprland" in result[0]["content"]

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Create a custom config
# config = DEFAULT_CONFIG.copy()
# config["llm_provider"] = "google"  # Use a different model
# config["backend_url"] = "https://generativelanguage.googleapis.com/v1"  # Use a different backend
# config["deep_think_llm"] = "gemini-2.0-flash"  # Use a different model
# config["quick_think_llm"] = "gemini-2.0-flash"  # Use a different model
# config["max_debate_rounds"] = 1  # Increase debate rounds
# config["online_tools"] = True  # Increase debate rounds

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"  # Use OpenAI
config["backend_url"] = "https://api.openai.com/v1"  # OpenAI API endpoint
config["deep_think_llm"] = "o4-mini"  # Use OpenAI's o4-mini model
config["quick_think_llm"] = "gpt-4o-mini"  # Use OpenAI's gpt-4o-mini model
config["max_debate_rounds"] = 1  # Increase debate rounds
config["online_tools"] = True  # Use online tools



# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("META", "2025-07-03")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns

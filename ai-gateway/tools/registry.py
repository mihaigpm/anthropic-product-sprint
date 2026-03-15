from typing import Dict, List, Any
from .base import BaseTool
from .stock_tool import StockPriceTool

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        # Self-register tools
        self.register(StockPriceTool())

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_definitions(self) -> List[dict]:
        return [tool.definition for tool in self._tools.values()]

    async def execute_tool(self, name: str, arguments: dict) -> Any:
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Tool {name} not found"}
        return await tool.execute(**arguments)

# Singleton instance
print("DEBUG: Reached the bottom of registry.py!")
registry = ToolRegistry()
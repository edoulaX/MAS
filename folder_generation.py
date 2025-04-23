import os

BASE = "agentic-research-assistant"

FOLDERS = [
    "agents",
    "tools",
    "memory",
    "workflow",
    "tests"
]

FILES = {
    "main.py": "",
    "config.py": "",
    "requirements.txt": "langgraph-swarm\n",
    "README.md": "#Agentic Research Assistant\n",
    # agents
    "agents/__init__.py": "",
    "agents/base.py": "",
    "agents/user_agent.py": "",
    "agents/paper_research_agent.py": "",
    "agents/summary_agent.py": "",
    # tools
    "tools/__init__.py": "",
    "tools/translator.py": "",
    "tools/mcp_arxiv.py": "",
    "tools/paper_ops.py": "",
    "tools/handoff.py": "",
    # memory
    "memory/__init__.py": "",
    "memory/short_term.py": "",
    "memory/long_term.py": "",
    # workflow
    "workflow/__init__.py": "",
    "workflow/state.py": "",
    "workflow/builder.py": "",
    "workflow/router.py": "",
    # tests
    "tests/test_agents.py": "",
    "tests/test_tools.py": "",
    "tests/test_workflow.py": ""
}


def scaffold_project():
    for folder in FOLDERS:
        os.makedirs(os.path.join(BASE, folder), exist_ok=True)

    for rel_path, content in FILES.items():
        abs_path = os.path.join(BASE, rel_path)
        with open(abs_path, "w") as f:
            f.write(content)

    print(f"✅ Project structure created in: ./{BASE}")


if __name__ == "__main__":
    scaffold_project()

"""Node-callable plain async functions.

These are called directly by graph node implementations — no LLM
decision involved. They return native Python types (dict, list),
not JSON strings. No @tool decorator.
"""

[Skip to main content](https://docs.crewai.com/en/changelog#content-area)

[CrewAI home page![light logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)![dark logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)](https://docs.crewai.com/)

![US](https://d3gk2c5xim1je2.cloudfront.net/flags/US.svg)

English

Search...

Ctrl K

Search...

Navigation

Release Notes

Changelog

[Home](https://docs.crewai.com/) [Documentation](https://docs.crewai.com/en/introduction) [AOP](https://docs.crewai.com/en/enterprise/introduction) [API Reference](https://docs.crewai.com/en/api-reference/introduction) [Examples](https://docs.crewai.com/en/examples/example) [Changelog](https://docs.crewai.com/en/changelog)

- [Website](https://crewai.com/)
- [Forum](https://community.crewai.com/)
- [Blog](https://blog.crewai.com/)
- [CrewGPT](https://chatgpt.com/g/g-qqTuUWsBY-crewai-assistant)

##### Release Notes

- [Changelog](https://docs.crewai.com/en/changelog)

[​](https://docs.crewai.com/en/changelog#sep-30%2C-2025)

Sep 30, 2025

## [​](https://docs.crewai.com/en/changelog\#v1-0-0a1)  v1.0.0a1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/1.0.0a1)

## [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed)  What’s Changed

### [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes)  Core Improvements & Fixes

- Fixed permission handling for `actions` in agent configuration
- Updated CI workflows and release publishing to support the new monorepo structure
- Bumped Python support to 3.13 and refreshed workspace metadata

### [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements)  New Features & Enhancements

- Added `apps` and `actions` attributes to `Agent` for richer runtime control
- Merged the `crewai-tools` repository into the main workspace (monorepo)
- Bumped all packages to 1.0.0a1 to mark the alpha milestone

### [​](https://docs.crewai.com/en/changelog\#cleanup-&-infrastructure)  Cleanup & Infrastructure

- Delivered a new CI pipeline with version pinning and publishing strategy
- Unified internal code to manage multiple packages coherently

[​](https://docs.crewai.com/en/changelog#sep-26%2C-2025)

Sep 26, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-201-1)  v0.201.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.201.1)

## [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-2)  What’s Changed

### [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-2)  Core Improvements & Fixes

- Renamed Watson embedding provider to `watsonx` and refreshed environment variable prefixes
- Added ChromaDB compatibility for `watsonx` and `voyageai` embedding providers

### [​](https://docs.crewai.com/en/changelog\#cleanup-&-deprecations)  Cleanup & Deprecations

- Standardized environment variable prefixes for all embedding providers
- Bumped CrewAI to 0.201.1 and updated internal dependencies

[​](https://docs.crewai.com/en/changelog#sep-24%2C-2025)

Sep 24, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-201-0)  v0.201.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.201.0)

## [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-3)  What’s Changed

### [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-3)  Core Improvements & Fixes

- Made the `ready` parameter optional in `_create_reasoning_plan`
- Fixed nested config handling for embedder configuration
- Added `batch_size` support to avoid token limit errors
- Corrected Quickstart documentation directory naming
- Resolved test duration cache issues and event exports
- Added fallback logic to crew settings

### [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-2)  New Features & Enhancements

- Introduced thread-safe platform context management
- Added `crewai uv` wrapper command to run `uv` from the CLI
- Enabled marking traces as failed for observability workflows
- Added custom embedding types and provider migration support
- Upgraded ChromaDB to v1.1.0 with compatibility fixes and type improvements
- Added Pydantic-compatible import validation and reorganized dependency groups

### [​](https://docs.crewai.com/en/changelog\#documentation-&-guides)  Documentation & Guides

- Updated changelog coverage for recent releases (0.193.x series)
- Documented metadata support for LLM Guardrail events
- Added guidance for fallback behavior and configuration visibility

### [​](https://docs.crewai.com/en/changelog\#cleanup-&-deprecations-2)  Cleanup & Deprecations

- Resolved Ruff and MyPy issues across modules
- Improved type annotations and consolidated utilities
- Deprecated legacy utilities in favor of Pydantic-compatible imports

### [​](https://docs.crewai.com/en/changelog\#contributors)  Contributors

- @qizwiz (first contribution)

[​](https://docs.crewai.com/en/changelog#sep-20%2C-2025)

Sep 20, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-193-2)  v0.193.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.193.2)

## [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-4)  What’s Changed

- Updated pyproject templates to use the right version

[​](https://docs.crewai.com/en/changelog#sep-20%2C-2025-2)

Sep 20, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-193-1)  v0.193.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.193.1)

## [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-5)  What’s Changed

- Series of minor fixes and linter improvements

[​](https://docs.crewai.com/en/changelog#sep-19%2C-2025)

Sep 19, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-193-0)  v0.193.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.193.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-4)  Core Improvements & Fixes

- Fixed handling of the `model` parameter during OpenAI adapter initialization
- Resolved test duration cache issues in CI workflows
- Fixed flaky test related to repeated tool usage by agents
- Added missing event exports to `__init__.py` for consistent module behavior
- Dropped message storage from metadata in Mem0 to reduce bloat
- Fixed L2 distance metric support for backward compatibility in vector search

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-3)  New Features & Enhancements

- Introduced thread-safe platform context management
- Added test duration caching for optimized `pytest-split` runs
- Added ephemeral trace improvements for better trace control
- Made search parameters for RAG, knowledge, and memory fully configurable
- Enabled ChromaDB to use OpenAI API for embedding functions
- Added deeper observability tools for user-level insights
- Unified RAG storage system with instance-specific client support

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-2)  Documentation & Guides

- Updated `RagTool` references to reflect CrewAI native RAG implementation
- Improved internal docs for `langgraph` and `openai` agent adapters with type annotations and docstrings

[​](https://docs.crewai.com/en/changelog#sep-11%2C-2025)

Sep 11, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-186-1)  v0.186.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.186.1)

## [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-6)  What’s Changed

- Fixed version not being found and silently failing reversion
- Bumped CrewAI version to 0.186.1 and updated dependencies in the CLI

[​](https://docs.crewai.com/en/changelog#sep-10%2C-2025)

Sep 10, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-186-0)  v0.186.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.186.0)

## [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-7)  What’s Changed

- Refer to the GitHub release notes for detailed changes

[​](https://docs.crewai.com/en/changelog#sep-04%2C-2025)

Sep 04, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-177-0)  v0.177.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.177.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-5)  Core Improvements & Fixes

- Achieved parity between `rag` package and current implementation
- Enhanced LLM event handling with task and agent metadata
- Fixed mutable default arguments by replacing them with `None`
- Suppressed Pydantic deprecation warnings during initialization
- Fixed broken example link in `README.md`
- Removed Python 3.12+ only Ruff rules for compatibility
- Migrated CI workflows to use `uv` and updated dev tooling

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-4)  New Features & Enhancements

- Added tracing improvements and cleanup
- Centralized event logic by moving `events` module to `crewai.events`

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-3)  Documentation & Guides

- Updated Enterprise Action Auth Token section documentation
- Published documentation updates for `v0.175.0` release

## [​](https://docs.crewai.com/en/changelog\#cleanup-&-refactoring)  Cleanup & Refactoring

- Refactored parser into modular functions for better structure

[​](https://docs.crewai.com/en/changelog#aug-28%2C-2025)

Aug 28, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-175-0)  v0.175.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.175.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-6)  Core Improvements & Fixes

- Fixed migration of the `tool` section during `crewai update`
- Reverted OpenAI pin: now requires `openai >=1.13.3` due to fixed import issues
- Fixed flaky tests and improved test stability
- Improved `Flow` listener resumability for HITL and cyclic flows
- Enhanced timeout handling in `PlusAPI` and `TraceBatchManager`
- Batched entity memory items to reduce redundant operations

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-5)  New Features & Enhancements

- Added support for additional parameters in `Flow.start()` methods
- Displayed task names in verbose CLI output
- Added centralized embedding types and introduced a base embedding client
- Introduced generic clients for ChromaDB and Qdrant
- Added support for `crewai config reset` to clear tokens
- Enabled `crewai_trigger_payload` auto-injection
- Simplified RAG client initialization and introduced RAG configuration system
- Added Qdrant RAG provider support
- Improved tracing with better event data
- Added support to remove Auth0 and email entry on `crewai login`

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-4)  Documentation & Guides

- Added documentation for automation triggers
- Fixed API Reference OpenAPI sources and redirects
- Added hybrid search alpha parameter to the docs

## [​](https://docs.crewai.com/en/changelog\#cleanup-&-deprecations-3)  Cleanup & Deprecations

- Added deprecation notice for `Task.max_retries`
- Removed Auth0 dependency from login flow

[​](https://docs.crewai.com/en/changelog#aug-19%2C-2025)

Aug 19, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-165-1)  v0.165.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.165.1)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-7)  Core Improvements & Fixes

- Fixed compatibility in `XMLSearchTool` by converting config values to strings for `configparser`
- Fixed flaky Pytest test involving `PytestUnraisableExceptionWarning`
- Mocked telemetry in test suite for more stable CI runs
- Moved Chroma lockfile handling to `db_storage_path`
- Ignored deprecation warnings from `chromadb`
- Pinned OpenAI version `<1.100.0` due to `ResponseTextConfigParam` import issue

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-6)  New Features & Enhancements

- Included exchanged agent messages into `ExternalMemory` metadata
- Automatically injected `crewai_trigger_payload`
- Renamed internal flag `inject_trigger_input` to `allow_crewai_trigger_context`
- Continued tracing improvements and ephemeral tracing logic
- Consolidated tracing logic conditions
- Added support for `agent_id`-linked memory entries in `Mem0`

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-5)  Documentation & Guides

- Added example to Tool Repository docs
- Updated Mem0 documentation for Short-Term and Entity Memory integration
- Revised Korean translations and improved sentence structures

## [​](https://docs.crewai.com/en/changelog\#cleanup-&-chores)  Cleanup & Chores

- Removed deprecated AgentOps integration

[​](https://docs.crewai.com/en/changelog#aug-19%2C-2025-2)

Aug 19, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-165-0)  v0.165.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.165.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-8)  Core Improvements & Fixes

- Fixed compatibility in `XMLSearchTool` by converting config values to strings for `configparser`
- Fixed flaky Pytest test involving `PytestUnraisableExceptionWarning`
- Mocked telemetry in test suite for more stable CI runs
- Moved Chroma lockfile handling to `db_storage_path`
- Ignored deprecation warnings from `chromadb`
- Pinned OpenAI version `<1.100.0` due to `ResponseTextConfigParam` import issue

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-7)  New Features & Enhancements

- Included exchanged agent messages into `ExternalMemory` metadata
- Automatically injected `crewai_trigger_payload`
- Renamed internal flag `inject_trigger_input` to `allow_crewai_trigger_context`
- Continued tracing improvements and ephemeral tracing logic
- Consolidated tracing logic conditions
- Added support for `agent_id`-linked memory entries in `Mem0`

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-6)  Documentation & Guides

- Added example to Tool Repository docs
- Updated Mem0 documentation for Short-Term and Entity Memory integration
- Revised Korean translations and improved sentence structures

## [​](https://docs.crewai.com/en/changelog\#cleanup-&-chores-2)  Cleanup & Chores

- Removed deprecated AgentOps integration

[​](https://docs.crewai.com/en/changelog#aug-13%2C-2025)

Aug 13, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-159-0)  v0.159.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.159.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-9)  Core Improvements & Fixes

- Improved LLM message formatting performance for better runtime efficiency
- Fixed use of incorrect endpoint in enterprise configuration auth/parameters
- Commented out listener resumability check for stability during partial flow resumption

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-8)  New Features & Enhancements

- Added `enterprise configure` command to CLI for streamlined enterprise setup
- Introduced partial flow resumability support

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-7)  Documentation & Guides

- Added documentation for new tools
- Added Korean translations
- Updated documentation with TrueFoundry integration details
- Added RBAC documentation and general cleanup
- Fixed API reference and revamped examples/cookbooks across EN, PT-BR, and KO

[​](https://docs.crewai.com/en/changelog#aug-06%2C-2025)

Aug 06, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-157-0)  v0.157.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.157.0)

## [​](https://docs.crewai.com/en/changelog\#v0-157-0-what%E2%80%99s-changed)  v0.157.0 What’s Changed

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-10)  Core Improvements & Fixes

- Enabled word wrapping for long input tool
- Allowed persisting Flow state with `BaseModel` entries
- Optimized string operations using `partition()` for performance
- Dropped support for deprecated User Memory system
- Bumped LiteLLM version to `1.74.9`
- Fixed CLI to show missing modules more clearly during import
- Supported device authorization with Okta

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-9)  New Features & Enhancements

- Added `crewai config` CLI command group with tests
- Added default value support for `crew.name`
- Introduced initial tracing capabilities
- Added support for LangDB integration
- Added support for CLI configuration documentation

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-8)  Documentation & Guides

- Updated MCP documentation with `connect_timeout` attribute
- Added LangDB integration documentation
- Added CLI config documentation
- General feature doc updates and cleanup

[​](https://docs.crewai.com/en/changelog#jul-30%2C-2025)

Jul 30, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-152-0)  v0.152.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.152.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-11)  Core Improvements & Fixes

- Removed `crewai signup` references and replaced them with `crewai login`
- Fixed support for adding memories to Mem0 using `agent_id`
- Changed the default value in Mem0 configuration
- Updated import error to show missing module files clearly
- Added timezone support to event timestamps

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-10)  New Features & Enhancements

- Enhanced `Flow` class to support custom flow names
- Refactored RAG components into a dedicated top-level module

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-9)  Documentation & Guides

- Fixed incorrect model naming in Google Vertex AI documentation

[​](https://docs.crewai.com/en/changelog#jul-23%2C-2025)

Jul 23, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-150-0)  v0.150.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.150.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-12)  Core Improvements & Fixes

- Used file lock around Chroma client initialization
- Removed workaround related to SQLite without FTS5
- Dropped unsupported `stop` parameter for LLM models automatically
- Fixed `save` method and updated related test cases
- Fixed message handling for Ollama models when last message is from assistant
- Removed duplicate print on LLM call error
- Added deprecation notice to `UserMemory`
- Upgraded LiteLLM to version 1.74.3

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-11)  New Features & Enhancements

- Added support for ad-hoc tool calling via internal LLM class
- Updated Mem0 Storage from v1.1 to v2

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-10)  Documentation & Guides

- Fixed neatlogs documentation
- Added Tavily Search & Extractor tools to the Search-Research suite
- Added documentation for `SerperScrapeWebsiteTool` and reorganized Serper section
- General documentation updates and improvements

## [​](https://docs.crewai.com/en/changelog\#crewai-tools-v0-58-0)  crewai-tools v0.58.0

### [​](https://docs.crewai.com/en/changelog\#new-tools-/-enhancements)  New Tools / Enhancements

- **SerperScrapeWebsiteTool**: Added a tool for extracting clean content from URLs
- **Bedrock AgentCore**: Integrated browser and code interpreter toolkits for Bedrock agents
- **Stagehand Update**: Refactored and updated Stagehand integration

### [​](https://docs.crewai.com/en/changelog\#fixes-&-cleanup)  Fixes & Cleanup

- **FTS5 Support**: Enabled SQLite FTS5 for improved text search in test workflows
- **Test Speedups**: Parallelized GitHub Actions test suite for faster CI runs
- **Cleanup**: Removed SQLite workaround due to FTS5 support being available
**MongoDBVectorSearchTool**: Fixed serialization and schema handling

[​](https://docs.crewai.com/en/changelog#jul-16%2C-2025)

Jul 16, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-148-0)  v0.148.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.148.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-13)  Core Improvements & Fixes

- Used production WorkOS environment ID
- Added SQLite FTS5 support to test workflow
- Fixed agent knowledge handling
- Compared using `BaseLLM` class instead of `LLM`
- Fixed missing `create_directory` parameter in `Task` class

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-12)  New Features & Enhancements

- Introduced Agent evaluation functionality
- Added Evaluator experiment and regression testing methods
- Implemented thread-safe `AgentEvaluator`
- Enabled event emission for Agent evaluation
- Supported evaluation of single `Agent` and `LiteAgent`
- Added integration with `neatlogs`
- Added crew context tracking for LLM guardrail events

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-11)  Documentation & Guides

- Added documentation for `guardrail` attributes and usage examples
- Added integration guide for `neatlogs`
- Updated documentation for Agent repository and `Agent.kickoff` usage

[​](https://docs.crewai.com/en/changelog#jul-09%2C-2025)

Jul 09, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-141-0)  v0.141.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.141.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-14)  Core Improvements & Fixes

- Sped up GitHub Actions tests through parallelization

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-13)  New Features & Enhancements

- Added crew context tracking for LLM guardrail events

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-12)  Documentation & Guides

- Added documentation for Agent repository usage
- Added documentation for `Agent.kickoff` method

[​](https://docs.crewai.com/en/changelog#jul-02%2C-2025)

Jul 02, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-140-0)  v0.140.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.140.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-15)  Core Improvements & Fixes

- Fixed typo in test prompts
- Fixed project name normalization by stripping trailing slashes during crew creation
- Ensured environment variables are written in uppercase
- Updated LiteLLM dependency
- Refactored collection handling in `RAGStorage`
- Implemented PEP 621 dynamic versioning

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-14)  New Features & Enhancements

- Added capability to track LLM calls by task and agent
- Introduced `MemoryEvents` to monitor memory usage
- Added console logging for memory system and LLM guardrail events
- Improved data training support for models up to 7B parameters
- Added Scarf and Reo.dev analytics tracking
- CLI workos login

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-13)  Documentation & Guides

- Updated CLI LLM documentation
- Added Nebius integration to the docs
- Corrected typos in installation and pt-BR documentation
- Added docs about `MemoryEvents`
- Implemented docs redirects and included development tools

[​](https://docs.crewai.com/en/changelog#jun-25%2C-2025)

Jun 25, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-134-0)  v0.134.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.134.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-16)  Core Improvements & Fixes

- Fixed tools parameter syntax
- Fixed type annotation in `Task`
- Fixed SSL error when retrieving LLM data from GitHub
- Ensured compatibility with Pydantic 2.7.x
- Removed `mkdocs` from project dependencies
- Upgraded Langfuse code examples to use Python SDK v3
- Added sanitize role feature in `mem0` storage
- Improved Crew search during memory reset
- Improved console printer output

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-15)  New Features & Enhancements

- Added support for initializing a tool from defined `Tool` attributes
- Added official way to use MCP Tools within a `CrewBase`
- Enhanced MCP tools support to allow selecting multiple tools per agent in `CrewBase`
- Added Oxylabs Web Scraping tools

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-14)  Documentation & Guides

- Updated `quickstart.mdx`
- Added docs on `LLMGuardrail` events
- Updated documentation with comprehensive service integration details
- Updated recommendation filters for MCP and Enterprise tools
- Updated docs for Maxim observability
- Added pt-BR documentation translation
- General documentation improvements

[​](https://docs.crewai.com/en/changelog#jun-12%2C-2025)

Jun 12, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-130-0)  v0.130.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.130.0)

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-17)  Core Improvements & Fixes

- Removed duplicated message related to Tool result output
- Fixed missing `manager_agent` tokens in `usage_metrics` from kickoff
- Fixed telemetry singleton to respect dynamic environment variables
- Fixed issue where Flow status logs could hide human input
- Increased default X-axis spacing for flow plotting

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-16)  New Features & Enhancements

- Added support for multi-org actions in the CLI
- Enabled async tool executions for more efficient workflows
- Introduced `LiteAgent` with Guardrail integration
- Upgraded `LiteLLM` to support latest OpenAI version

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-15)  Documentation & Guides

- Documented minimum `UV` version for Tool repository
- Improved examples for Hallucination Guardrail
- Updated planning docs for LLM usage
- Added documentation for Maxim support in Agent observability
- Expanded integrations documentation with images for enterprise features
- Fixed guide on persistence
- Updated Python version support to support python 3.13.x

[​](https://docs.crewai.com/en/changelog#jun-05%2C-2025)

Jun 05, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-126-0)  v0.126.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.126.0)

### [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-8)  What’s Changed

#### [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-18)  Core Improvements & Fixes

- Added support for Python 3.13
- Fixed agent knowledge sources issue
- Persisted available tools from a Tool repository
- Enabled tools to be loaded from Agent repository via their own module
- Logged usage of tools when called by an LLM

#### [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-17)  New Features & Enhancements

- Added streamable-http transport support in MCP integration
- Added support for community analytics
- Expanded OpenAI-compatible section with a Gemini example
- Introduced transparency features for prompts and memory systems
- Minor enhancements for Tool publishing

#### [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-16)  Documentation & Guides

- Major restructuring of docs for better navigation
- Expanded MCP integration documentation
- Updated memory docs and README visuals
- Fixed missing await keywords in async kickoff examples
- Updated Portkey and Azure embeddings documentation
- Added enterprise testing image to the LLM guide
- General updates to the README

[​](https://docs.crewai.com/en/changelog#may-27%2C-2025)

May 27, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-121-1)  v0.121.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.121.1)Bug fixes and better docs

[​](https://docs.crewai.com/en/changelog#may-22%2C-2025)

May 22, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-121-0)  v0.121.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.121.0)

# [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-9)  What’s Changed

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-19)  Core Improvements & Fixes

- Fixed encoding error when creating tools
- Fixed failing llama test
- Updated logging configuration for consistency
- Enhanced telemetry initialization and event handling

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-18)  New Features & Enhancements

- Added markdown attribute to the Task class
- Added reasoning attribute to the Agent class
- Added inject\_date flag to Agent for automatic date injection
- Implemented HallucinationGuardrail (no-op with test coverage)

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-17)  Documentation & Guides

- Added documentation for StagehandTool and improved MDX structure
- Added documentation for MCP integration and updated enterprise docs
- Documented knowledge events and updated reasoning docs
- Added stop parameter documentation
- Fixed import references in doc examples (before\_kickoff, after\_kickoff)
- General docs updates and restructuring for clarity

[​](https://docs.crewai.com/en/changelog#may-15%2C-2025)

May 15, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-120-1)  v0.120.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.120.1)

## [​](https://docs.crewai.com/en/changelog\#whats-new)  Whats New

- Fixes Interpolation with hyphens

[​](https://docs.crewai.com/en/changelog#may-14%2C-2025)

May 14, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-120-0)  v0.120.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.120.0)

### [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-20)  Core Improvements & Fixes

• Enabled full Ruff rule set by default for stricter linting
• Addressed race condition in FilteredStream using context managers
• Fixed agent knowledge reset issue
• Refactored agent fetching logic into utility module

### [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-19)  New Features & Enhancements

• Added support for loading an Agent directly from a repository
• Enabled setting an empty context for Task
• Enhanced Agent repository feedback and fixed Tool auto-import behavior
• Introduced direct initialization of knowledge (bypassing knowledge\_sources)

### [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-18)  Documentation & Guides

• Updated security.md for current security practices
• Cleaned up Google setup section for clarity
• Added link to AI Studio when entering Gemini key
• Updated Arize Phoenix observability guide
• Refreshed flow documentation

[​](https://docs.crewai.com/en/changelog#may-08%2C-2025)

May 08, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-119-0)  v0.119.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.119.0)What’s Changed

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-21)  Core Improvements & Fixes

- Improved test reliability by enhancing pytest handling for flaky tests
- Fixed memory reset crash when embedding dimensions mismatch
- Enabled parent flow identification for Crew and LiteAgent
- Prevented telemetry-related crashes when unavailable
- Upgraded LiteLLM version for better compatibility
- Fixed llama converter tests by removing skip\_external\_api

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-20)  New Features & Enhancements

- Introduced knowledge retrieval prompt re-writting in Agent for improved tracking and debugging
- Made LLM setup and quickstart guides model-agnostic

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-19)  Documentation & Guides

- Added advanced configuration docs for the RAG tool
- Updated Windows troubleshooting guide
- Refined documentation examples for better clarity
- Fixed typos across docs and config files

[​](https://docs.crewai.com/en/changelog#apr-30%2C-2025)

Apr 30, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-118-0)  v0.118.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.118.0)

### [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-22)  Core Improvements & Fixes

- Fixed issues with missing prompt or system templates.
- Removed global logging configuration to avoid unintended overrides.
- Renamed TaskGuardrail to LLMGuardrail for improved clarity.
- Downgraded litellm to version 1.167.1 for compatibility.
- Added missing **init**.py files to ensure proper module initialization.

### [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-21)  New Features & Enhancements

- Added support for no-code Guardrail creation to simplify AI behavior controls.

### [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-20)  Documentation & Guides

- Removed CrewStructuredTool from public documentation to reflect internal usage.
- Updated enterprise documentation and YouTube embed for improved onboarding experience.

[​](https://docs.crewai.com/en/changelog#apr-28%2C-2025)

Apr 28, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-117-1)  v0.117.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.117.1)

- build: upgrade crewai-tools
- upgrade liteLLM to latest version
- Fix Mem0 OSS

[​](https://docs.crewai.com/en/changelog#apr-28%2C-2025-2)

Apr 28, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-117-0)  v0.117.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.117.0)

# [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-10)  What’s Changed

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-22)  New Features & Enhancements

- Added `result_as_answer` parameter support in `@tool` decorator.
- Introduced support for new language models: GPT-4.1, Gemini-2.0, and Gemini-2.5 Pro.
- Enhanced knowledge management capabilities.
- Added Huggingface provider option in CLI.
- Improved compatibility and CI support for Python 3.10+.

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-23)  Core Improvements & Fixes

- Fixed issues with incorrect template parameters and missing inputs.
- Improved asynchronous flow handling with coroutine condition checks.
- Enhanced memory management with isolated configuration and correct memory object copying.
- Fixed initialization of lite agents with correct references.
- Addressed Python type hint issues and removed redundant imports.
- Updated event placement for improved tool usage tracking.
- Raised explicit exceptions when flows fail.
- Removed unused code and redundant comments from various modules.
- Updated GitHub App token action to v2.

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-21)  Documentation & Guides

- Enhanced documentation structure, including enterprise deployment instructions.
- Automatically create output folders for documentation generation.
- Fixed broken link in `WeaviateVectorSearchTool` documentation.
- Fixed guardrail documentation usage and import paths for JSON search tools.
- Updated documentation for `CodeInterpreterTool`.
- Improved SEO, contextual navigation, and error handling for documentation pages.

[​](https://docs.crewai.com/en/changelog#apr-10%2C-2025)

Apr 10, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-114-0)  v0.114.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.114.0)

# [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-changed-11)  What’s Changed

## [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-23)  New Features & Enhancements

- Agents as an atomic unit. (`Agent(...).kickoff()`)
- Support to Custom LLM implementations.
- Integrated External Memory and Opik observability.
- Enhanced YAML extraction.
- Multimodal agent validation.
- Added Secure fingerprints for agents and crews.

## [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-24)  Core Improvements & Fixes

- Improved serialization, agent copying, and Python compatibility.
- Added wildcard support to emit()
- Added support for additional router calls and context window adjustments.
- Fixed typing issues, validation, and import statements.
- Improved method performance.
- Enhanced agent task handling, event emissions, and memory management.
- Fixed CLI issues, conditional tasks, cloning behavior, and tool outputs.

## [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-22)  Documentation & Guides

- Improved documentation structure, theme, and organization.
- Added guides for Local NVIDIA NIM with WSL2, W&B Weave, and Arize Phoenix.
- Updated tool configuration examples, prompts, and observability docs.
- Guide on using singular agents within Flows

[​](https://docs.crewai.com/en/changelog#mar-17%2C-2025)

Mar 17, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-108-0)  v0.108.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.108.0)

# [​](https://docs.crewai.com/en/changelog\#features)  Features

- Converted tabs to spaces in crew.py template in PR #2190
- Enhanced LLM Streaming Response Handling and Event System in PR #2266
- Included model\_name in PR #2310
- Enhanced Event Listener with rich visualization and improved logging in PR #2321
- Added fingerprints in PR #2332

# [​](https://docs.crewai.com/en/changelog\#bug-fixes)  Bug Fixes

- Fixed Mistral issues in PR #2308
- Fixed a bug in documentation in PR #2370
- Fixed type check error in fingerprint property in PR #2369

# [​](https://docs.crewai.com/en/changelog\#documentation-updates)  Documentation Updates

- Improved tool documentation in PR #2259
- Updated installation guide for the uv tool package in PR #2196
- Added instructions for upgrading crewAI with the uv tool in PR #2363
- Added documentation for ApifyActorsTool in PR #2254

[​](https://docs.crewai.com/en/changelog#mar-09%2C-2025)

Mar 09, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-105-0)  v0.105.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.105.0)**Core Improvements & Fixes**

- Fixed issues with missing template variables and user memory configuration.
- Improved async flow support and addressed agent response formatting.
- Enhanced memory reset functionality and fixed CLI memory commands.
- Fixed type issues, tool calling properties, and telemetry decoupling.

**New Features & Enhancements**

- Added Flow state export and improved state utilities.
- Enhanced agent knowledge setup with optional crew embedder.
- Introduced event emitter for better observability and LLM call tracking.
- Added support for Python 3.10 and ChatOllama from langchain\_ollama.
- Integrated context window size support for the o3-mini model.
- Added support for multiple router calls.

**Documentation & Guides**

- Improved documentation layout and hierarchical structure.
- Added QdrantVectorSearchTool guide and clarified event listener usage.
- Fixed typos in prompts and updated Amazon Bedrock model listings.

[​](https://docs.crewai.com/en/changelog#feb-13%2C-2025)

Feb 13, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-102-0)  v0.102.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.102.0)

### [​](https://docs.crewai.com/en/changelog\#core-improvements-&-fixes-25)  Core Improvements & Fixes

- Enhanced LLM Support: Improved structured LLM output, parameter handling, and formatting for Anthropic models.
- Crew & Agent Stability: Fixed issues with cloning agents/crews using knowledge sources, multiple task outputs in conditional tasks, and ignored Crew task callbacks.
- Memory & Storage Fixes: Fixed short-term memory handling with Bedrock, ensured correct embedder initialization, and added a reset memories function in the crew class.
- Training & Execution Reliability: Fixed broken training and interpolation issues with dict and list input types.

### [​](https://docs.crewai.com/en/changelog\#new-features-&-enhancements-24)  New Features & Enhancements

- Advanced Knowledge Management: Improved naming conventions and enhanced embedding configuration with custom embedder support.
- Expanded Logging & Observability: Added JSON format support for logging and integrated MLflow tracing documentation.
- Data Handling Improvements: Updated excel\_knowledge\_source.py to process multi-tab files.
- General Performance & Codebase Clean-Up: Streamlined enterprise code alignment and resolved linting issues.
- Adding new tool QdrantVectorSearchTool

### [​](https://docs.crewai.com/en/changelog\#documentation-&-guides-23)  Documentation & Guides

- Updated AI & Memory Docs: Improved Bedrock, Google AI, and long-term memory documentation.
- Task & Workflow Clarity: Added “Human Input” row to Task Attributes, Langfuse guide, and FileWriterTool documentation.
- Fixed Various Typos & Formatting Issues.

### [​](https://docs.crewai.com/en/changelog\#maintenance-&-miscellaneous)  Maintenance & Miscellaneous

- Refined Google Docs integrations and task handling for the current year.

[​](https://docs.crewai.com/en/changelog#jan-28%2C-2025)

Jan 28, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-100-0)  v0.100.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.100.0)

- Feat: Add Composio docs
- Feat: Add SageMaker as a LLM provider
- Fix: Overall LLM connection issues
- Fix: Using safe accessors on training
- Fix: Add version check to crew\_chat.py
- Docs: New docs for crewai chat
- Docs: Improve formatting and clarity in CLI and Composio Tool docs

[​](https://docs.crewai.com/en/changelog#jan-20%2C-2025)

Jan 20, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-98-0)  v0.98.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.98.0)

- Feat: Conversation crew v1
- Feat: Add unique ID to flow states
- Feat: Add @persist decorator with FlowPersistence interface
- Integration: Add SambaNova integration
- Integration: Add NVIDIA NIM provider in cli
- Integration: Introducing VoyageAI
- Chore: Update date to current year in template
- Fix: Fix API Key Behavior and Entity Handling in Mem0 Integration
- Fix: Fixed core invoke loop logic and relevant tests
- Fix: Make tool inputs actual objects and not strings
- Fix: Add important missing parts to creating tools
- Fix: Drop litellm version to prevent windows issue
- Fix: Before kickoff if inputs are none
- Fix: TYPOS
- Fix: Nested pydantic model issue
- Fix: Docling issues
- Fix: union issue
- Docs updates

[​](https://docs.crewai.com/en/changelog#jan-04%2C-2025)

Jan 04, 2025

## [​](https://docs.crewai.com/en/changelog\#v0-95-0)  v0.95.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.95.0)

- Feat: Adding Multimodal Abilities to Crew
- Feat: Programatic Guardrails
- Feat: HITL multiple rounds
- Feat: Gemini 2.0 Support
- Feat: CrewAI Flows Improvements
- Feat: Add Workflow Permissions
- Feat: Add support for langfuse with litellm
- Feat: Portkey Integration with CrewAI
- Feat: Add interpolate\_only method and improve error handling
- Feat: Docling Support
- Feat: Weviate Support
- Fix: output\_file not respecting system path
- Fix disk I/O error when resetting short-term memory.
- Fix: CrewJSONEncoder now accepts enums
- Fix: Python max version
- Fix: Interpolation for output\_file in Task
- Fix: Handle coworker role name case/whitespace properly
- Fix: Add tiktoken as explicit dependency and document Rust requirement
- Fix: Include agent knowledge in planning process
- Fix: Change storage initialization to None for KnowledgeStorage
- Fix: Fix optional storage checks
- Fix: include event emitter in flows
- Fix: Docstring, Error Handling, and Type Hints Improvements
- Fix: Suppressed userWarnings from litellm pydantic issues

[​](https://docs.crewai.com/en/changelog#dec-05%2C-2024)

Dec 05, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-86-0)  v0.86.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.86.0)

- remove all references to pipeline and pipeline router
- docs: Add Nvidia NIM as provider in Custom LLM
- add knowledge demo + improve knowledge docs
- Brandon/cre 509 hitl multiple rounds of followup
- New docs about yaml crew with decorators. Simplify template crew

[​](https://docs.crewai.com/en/changelog#dec-04%2C-2024)

Dec 04, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-85-0)  v0.85.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.85.0)

- Added knowledge to agent level
- Feat/remove langchain
- Improve typed task outputs
- Log in to Tool Repository on `crewai login`
- Fixes issues with result as answer not properly exiting LLM loop
- fix: missing key name when running with ollama provider
- fix spelling issue found
- Update readme for running mypy
- Add knowledge to mint.json
- Update Github actions
- Docs Update Agents docs to include two approaches for creating an agent
- Documentation Improvements: LLM Configuration and Usage

[​](https://docs.crewai.com/en/changelog#nov-25%2C-2024)

Nov 25, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-83-0)  v0.83.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.83.0)

- New `before_kickoff` and `after_kickoff` crew callbacks
- Support to pre-seed agents with Knowledge
- Add support for retrieving user preferences and memories using Mem0
- Fix Async Execution
- Upgrade chroma and adjust embedder function generator
- Update CLI Watson supported models + docs
- Reduce level for Bandit
- Fixing all tests
- Update Docs

[​](https://docs.crewai.com/en/changelog#nov-14%2C-2024)

Nov 14, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-80-0)  v0.80.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.80.0)

- Fixing Tokens callback replacement bug
- Fixing Step callback issue
- Add cached prompt tokens info on usage metrics
- Fix crew\_train\_success test

[​](https://docs.crewai.com/en/changelog#nov-11%2C-2024)

Nov 11, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-79-4)  v0.79.4

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.79.4)Series of small bug fixes around llms support

[​](https://docs.crewai.com/en/changelog#nov-10%2C-2024)

Nov 10, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-79-0)  v0.79.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.79.0)

- Add inputs to flows
- Enhance log storage to support more data types
- Add support to IBM memory
- Add Watson as an option in CLI
- Add security.md file
- Replace .netrc with uv environment variables
- Move BaseTool to main package and centralize tool description generation
- Raise an error if an LLM doesnt return a response
- Fix flows to support cycles and added in test
- Update how we name crews and fix missing config
- Update docs

[​](https://docs.crewai.com/en/changelog#oct-30%2C-2024)

Oct 30, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-76-9)  v0.76.9

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.76.9)

- Update plot command for flow to crewai flow plot
- Add tomli so we can support 3.10
- Forward install command options to `uv sync`
- Improve tool text description and args
- Improve tooling and flow docs
- Update flows cli to allow you to easily add additional crews to a flow with crewai flow add-crew
- Fixed flows bug when using multiple start and listen(and\_(…, …, …))

[​](https://docs.crewai.com/en/changelog#oct-23%2C-2024)

Oct 23, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-76-2)  v0.76.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.76.2)Updating crewai create commadn

[​](https://docs.crewai.com/en/changelog#oct-23%2C-2024-2)

Oct 23, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-76-0)  v0.76.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.76.0)

- fix/fixed missing API prompt + CLI docs update
- chore(readme): fixing step for ‘running tests’ in the contribution
- support unsafe code execution. add in docker install and running checks
- Fix memory imports for embedding functions by

[​](https://docs.crewai.com/en/changelog#oct-23%2C-2024-3)

Oct 23, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-75-1)  v0.75.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.75.1)new `--provider` option on crewai crewat

[​](https://docs.crewai.com/en/changelog#oct-23%2C-2024-4)

Oct 23, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-75-0)  v0.75.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.75.0)

- Fixing test post training
- Simplify flows
- Adapt `crewai tool install <tool>`
- Ensure original embedding config works
- Fix bugs
- Update docs - Including adding Cerebras LLM example configuration to LLM docs
- Drop unnecessary tests

[​](https://docs.crewai.com/en/changelog#oct-18%2C-2024)

Oct 18, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-74-2)  v0.74.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.74.2)

- feat: add poetry.lock to uv migration
- fix tool calling issue

[​](https://docs.crewai.com/en/changelog#oct-18%2C-2024-2)

Oct 18, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-74-0)  v0.74.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.74.0)

- UV migration
- Adapt Tools CLI to UV
- Add warning from Poetry -> UV
- CLI to allow for model selection & submitting API keys
- New Memory Base
- Fix Linting and Warnings
- Update Docs
- Bug fixesh

[​](https://docs.crewai.com/en/changelog#oct-11%2C-2024)

Oct 11, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-70-1)  v0.70.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.70.1)

- New Flow feature
- Flow visualizer
- Create `crewai create flow` command
- Create `crewai tool create <tool>` command
- Add Git validations for publishing tools
- fix: JSON encoding date objects
- New Docs
- Update README
- Bug fixes

[​](https://docs.crewai.com/en/changelog#sep-27%2C-2024)

Sep 27, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-65-2)  v0.65.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.65.2)

- Adding experimental Flows feature
- Fixing order of tasks bug
- Updating templates

[​](https://docs.crewai.com/en/changelog#sep-27%2C-2024-2)

Sep 27, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-64-0)  v0.64.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.64.0)

- Ordering tasks properly
- Fixing summarization logic
- Fixing stop words logic
- Increases default max iterations to 20
- Fix crew’s key after input interpolation
- Fixing Training Feature
- Adding initial tools API
- TYPOS
- Updating Docs

Fixes: #1359 #1355 #1353 #1356 and others

[​](https://docs.crewai.com/en/changelog#sep-25%2C-2024)

Sep 25, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-63-6)  v0.63.6

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.63.6)

- Updating projects templates

[​](https://docs.crewai.com/en/changelog#sep-25%2C-2024-2)

Sep 25, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-63-5)  v0.63.5

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.63.5)

- Bringing support to o1 family back, and any model that don’t support stop words
- Updating dependencies
- Updating logs
- Updating docs

[​](https://docs.crewai.com/en/changelog#sep-24%2C-2024)

Sep 24, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-63-2)  v0.63.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.63.2)

- Adding OPENAI\_BASE\_URL as fallback
- Adding proper LLM import
- Updating docs

[​](https://docs.crewai.com/en/changelog#sep-24%2C-2024-2)

Sep 24, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-63-1)  v0.63.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.63.1)

- Small bug fix for support future CrewAI deploy

[​](https://docs.crewai.com/en/changelog#sep-24%2C-2024-3)

Sep 24, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-63-0)  v0.63.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.63.0)

- New LLM class to interact with LLMs (leveraging LiteLLM)
- Adding support to custom memory interfaces
- Bringing GPT-4o-mini as the default model
- Updates Docs
- Updating dependencies
- Bug fixes
  - Remove redundant task creation in `kickoff_for_each_async`

[​](https://docs.crewai.com/en/changelog#sep-18%2C-2024)

Sep 18, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-61-0)  v0.61.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.61.0)

- Updating dependencies
- Printing max rpm message in different color
- Updating all cassettes for tests
- Always ending on a user message - to better support certain models like bedrock ones
- Overall small bug fixes

[​](https://docs.crewai.com/en/changelog#sep-16%2C-2024)

Sep 16, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-60-0)  v0.60.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.60.0)

- Removing LangChain and Rebuilding Executor
- Get all of out tests back to green
- Adds the ability to not use system prompt use\_system\_prompt on the Agent
- Adds the ability to not use stop words (to support o1 models) use\_stop\_words on the Agent
- Sliding context window gets renamed to respect\_context\_window, and enable by default
- Delegation is now disabled by default
- Inner prompts were slightly changed as well
- Overall reliability and quality of results
- New support for:
  - Number of max requests per minute
  - A maximum number of iterations before giving a final answer
  - Proper take advantage of system prompts
  - Token calculation flow
  - New logging of the crew and agent execution

[​](https://docs.crewai.com/en/changelog#sep-13%2C-2024)

Sep 13, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-55-2)  v0.55.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.55.2)

- Adding ability for auto complete
- Add name and expected\_output to TaskOutput
- New `crewai install` CLI
- New `crewai deploy` CLI
- Cleaning up of Pipeline feature
- Updated docs
- Dev experience improvements like bandit CI pipeline
- Fix bugs:
  - Ability to use `planning_llm`
  - Fix YAML based projects
  - Fix Azure support
  - Add support to Python 3.10
  - Moving away from Pydantic v1

[​](https://docs.crewai.com/en/changelog#aug-11%2C-2024)

Aug 11, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-51-0)  v0.51.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.51.0)

- crewAI Testing / Evaluation - [https://docs.crewai.com/core-concepts/Testing/](https://docs.crewai.com/core-concepts/Testing/)
- Adding new sliding context window
- Allowing all attributes on YAML - [https://docs.crewai.com/getting-started/Start-a-New-CrewAI-Project-Template-Method/#customizing-your-project](https://docs.crewai.com/getting-started/Start-a-New-CrewAI-Project-Template-Method/#customizing-your-project)
- Adding initial Pipeline Structure - [https://docs.crewai.com/core-concepts/Pipeline/](https://docs.crewai.com/core-concepts/Pipeline/)
- Ability to set LLM for planning step - [https://docs.crewai.com/core-concepts/Planning/](https://docs.crewai.com/core-concepts/Planning/)
- New crew run command - [https://docs.crewai.com/getting-started/Start-a-New-CrewAI-Project-Template-Method/#running-your-project](https://docs.crewai.com/getting-started/Start-a-New-CrewAI-Project-Template-Method/#running-your-project)
- Saving file now dumps dict into JSON - [https://docs.crewai.com/core-concepts/Tasks/#creating-directories-when-saving-files](https://docs.crewai.com/core-concepts/Tasks/#creating-directories-when-saving-files)
- Using verbose settings for tool outputs
- Added new Github Templates
- New Vision tool - [https://docs.crewai.com/tools/VisionTool/](https://docs.crewai.com/tools/VisionTool/)
- New DALL-E Tool - [https://docs.crewai.com/tools/DALL-ETool/](https://docs.crewai.com/tools/DALL-ETool/)
- New MySQL tool - [https://docs.crewai.com/tools/MySQLTool/](https://docs.crewai.com/tools/MySQLTool/)
- New NL2SQL Tool - [https://docs.crewai.com/tools/NL2SQLTool.md](https://docs.crewai.com/tools/NL2SQLTool.md)
- Bug Fixes:
  - Bug with planning feature output
  - Async tasks for hierarchical process
  - Better pydantic output for non OAI models
  - JSON truncation issues
  - Fix logging types
  - Only import AgentOps if the Env Key is set
  - Sanitize agent roles to ensure valid directory names (Windows)
  - Tools name shouldn’t contain space for OpenAI
  - A bunch of minor issues

[​](https://docs.crewai.com/en/changelog#jul-20%2C-2024)

Jul 20, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-41-1)  v0.41.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.41.1)

- Fix bug with planning feature

[​](https://docs.crewai.com/en/changelog#jul-19%2C-2024)

Jul 19, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-41-0)  v0.41.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.41.0)

- **\[Breaking Change\]** Type Safe output

  - All crews and tasks now return a proper object TaskOuput and CrewOutput
- **\[Feature\]** New planning feature for crews (plan before act)

  - by adding planning=True to the Crew instance
- **\[Feature\]** Introduced Replay Feature

  - New CLI that allow you to list the tasks from last run and replay from a specific one
- **\[Feature\]** Ability to reset memory

  - You can clean your crew memory before running it again
- **\[Feature\]** Add retry feature for LLM calls

  - You can retry llm calls and not stop the crew execution
- **\[Feature\]** Added ability to customize converter
- **\[Tool\]** Enhanced tools with type hinting and new attributes
- **\[Tool\]** Added MultiON Tool
- **\[Tool\]** Fixed filecrawl tools
- **\[Tool\]** Fixed bug in Scraping tool
- **\[Tools\]** Bumped crewAI-tools dependency to version
- **\[Bugs\]** General bug fixes and improvements
- **\[Bugs\]** Telemetry fixes
- **\[Bugs\]** Spell check corrections
- **\[Docs\]** Updated documentation

[​](https://docs.crewai.com/en/changelog#jul-06%2C-2024)

Jul 06, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-36-0)  v0.36.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.36.0)

- Bug fix
- Updating Docs
- Updating native prompts
- Fixing TYPOs on the prompts
- Adding AgentOps native support
- Adding Firecrawl Tools
- Adding new ability to return a tool results as an agent result
- Improving coding Interpreter tool
- Adding new option to create your own corveter class (docs pending)

[​](https://docs.crewai.com/en/changelog#jul-04%2C-2024)

Jul 04, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-35-8)  v0.35.8

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.35.8)

- fixing embechain dependency issue

[​](https://docs.crewai.com/en/changelog#jul-02%2C-2024)

Jul 02, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-35-7)  v0.35.7

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.35.7)

- New @composiohq integration is out
- Documentation update
- Custom GPT Updated
- Adjusting manager verbosity level
- Bug fixes

[​](https://docs.crewai.com/en/changelog#jul-01%2C-2024)

Jul 01, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-35-5)  v0.35.5

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.35.5)

- Fix embedchain dependency

[​](https://docs.crewai.com/en/changelog#jul-01%2C-2024-2)

Jul 01, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-35-4)  v0.35.4

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.35.4)

- Updating crewai create CLI to use the new version

[​](https://docs.crewai.com/en/changelog#jul-01%2C-2024-3)

Jul 01, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-35-3)  v0.35.3

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.35.3)

- Code Execution Bug fixed
- Updating overall docs
- Bumping version of crewai-tools
- Bumping versions of many dependencies
- Overall bugfixes

[​](https://docs.crewai.com/en/changelog#jun-29%2C-2024)

Jun 29, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-35-0)  v0.35.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.35.0)

- Your agents can now execute code
- Bring Any 3rd-party agent, LlamaIndex, LangChain and Autogen agents can all be part of your crew now!
- Train you crew before you execute it and get consistent results! New CLI `crewai train -n X`
- Bug fixes and docs updates (still missing some new docs updates coming soon)

[​](https://docs.crewai.com/en/changelog#jun-22%2C-2024)

Jun 22, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-32-2)  v0.32.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.32.2)

- Updating `crewai create` CLI to use the new version
- Fixing delegation agent matching

[​](https://docs.crewai.com/en/changelog#jun-21%2C-2024)

Jun 21, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-32-0)  v0.32.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.32.0)

- New `kickoff_for_each`, `kickoff_async` and `kickoff_for_each_async` methods for better control over the kickoff process
- Adding support for all LlamaIndex hub integrations
- Adding `usage_metrics` to full output or a crew
- Adding support to multiple crews on the new YAML format
- Updating dependencies
- Fixed Bugs and TYPOs
- Documentation updated
- Added search in docs
- Making gpt-4o the default model
- Adding new docs for LangTrace, Browserbase and Exa Search
- Adding timestamp to logging

[​](https://docs.crewai.com/en/changelog#may-23%2C-2024)

May 23, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-30-11)  v0.30.11

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.30.11)

- Updating project generation template

[​](https://docs.crewai.com/en/changelog#may-14%2C-2024)

May 14, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-30-8)  v0.30.8

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.30.8)

- Updating dependencies
- Small bug fixes on crewAI project structure
- Removing custom YAML parser for now

[​](https://docs.crewai.com/en/changelog#may-14%2C-2024-2)

May 14, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-30-5)  v0.30.5

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.30.5)

- Making agent delegation more versatile for smaller models

[​](https://docs.crewai.com/en/changelog#may-13%2C-2024)

May 13, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-30-4)  v0.30.4

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.30.4)**Docs Update will follow** sorry about that and thank you for bearing with me, we are launching new docs soon!➿ Fixing task callback
🧙 Ability to set a specific agent as manager instead of having crew create your one
📄 Ability to set system, prompt and response templates, so it works more reliable with opensource models (works better with smaller models)
👨‍💻 Improving json and pydantic output (works better with smaller models)
🔎 Improving tool name recognition (works better with smaller models)
🧰 Improvements for tool usage (works better with smaller models)
📃 Initial support to bring your own prompts
2️⃣ Fixing duplicating token calculator metrics
🪚 Adding couple new tools, Browserbase and Exa Search
📁 Ability to create directory when saving as file
🔁 Updating dependencies - double check tools
📄 Overall small documentation improvements
🐛 Smaller bug fixes (typos and such)
👬 Fixing co-worker / coworker issues
👀 Smaller Readme Updates

[​](https://docs.crewai.com/en/changelog#apr-11%2C-2024)

Apr 11, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-28-8)  v0.28.8

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.28.8)

- updating version used on crewai CLI

[​](https://docs.crewai.com/en/changelog#apr-11%2C-2024-2)

Apr 11, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-28-7)  v0.28.7

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/0.28.7)

- Bug fixes
- Updating crewAI tool version with bug fixes

[​](https://docs.crewai.com/en/changelog#apr-08%2C-2024)

Apr 08, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-28-5)  v0.28.5

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.28.5)

- Major Long term memory interpolation issue
- Updating tools package dependency with fixes
- Removing unnecessary certificate

[​](https://docs.crewai.com/en/changelog#apr-07%2C-2024)

Apr 07, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-28-2)  v0.28.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.28.2)

- Major long term memory fix

[​](https://docs.crewai.com/en/changelog#apr-06%2C-2024)

Apr 06, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-28-1)  v0.28.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.28.1)

- Updating crewai-tools to 0.1.15

[​](https://docs.crewai.com/en/changelog#apr-05%2C-2024)

Apr 05, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-28-0)  v0.28.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.28.0)

- Not overriding LLM callbacks
- Adding `max_execution_time` support
- Adding specific memory docs
- Moving tool usage logging color to purple from yellow
- Updating Docs

[​](https://docs.crewai.com/en/changelog#apr-04%2C-2024)

Apr 04, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-27-0)  v0.27.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.27.0)

- 🧠 **Memory (shared crew memory)** \- To enable it just add `memory=True` to your crew, it will work transparently and make outcomes better and more reliable, it’s disable by default for now
- 🤚🏼 **Native Human Input Support:** [docs](https://docs.crewai.com/how-to/Human-Input-on-Execution/)
- 🌐 **Universal RAG Tools Support:** Any models, beyond just OpenAI. [Example](https://docs.crewai.com/tools/DirectorySearchTool/#custom-model-and-embeddings)
- 🔍 **Enhanced Cache Control:** Meet the ingenious cache\_function attribute: [docs](https://docs.crewai.com/core-concepts/Tools/#custom-caching-mechanism)
- 🔁 **Updated crewai-tools Dependency:** Always in sync with the latest and greatest.
- ⛓️ **Cross Agent Delegation:** Smoother cooperation between agents.
- 💠 **Inner Prompt Improvements:** A finer conversational flow.
- 📝 **Improving tool usage with better parsing**
- 🔒 **Security improvements and updating dependencies**
- 📄 **Documentation improved**
- 🐛 **Bug fixes**

[​](https://docs.crewai.com/en/changelog#mar-12%2C-2024)

Mar 12, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-22-5)  v0.22.5

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.22.5)

- Other minor import issues on the new templates

[​](https://docs.crewai.com/en/changelog#mar-12%2C-2024-2)

Mar 12, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-22-4)  v0.22.4

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.22.4)Fixing template issues

[​](https://docs.crewai.com/en/changelog#mar-11%2C-2024)

Mar 11, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-22-2)  v0.22.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.22.2)

- Fixing bug on the new cli template
- Guaranteeing tasks order on new cli template

[​](https://docs.crewai.com/en/changelog#mar-11%2C-2024-2)

Mar 11, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-22-0)  v0.22.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.22.0)

- Adding initial CLI `crewai create` command
- Adding ability to agents and tasks to be defined using dictionaries
- Adding more clear agent logging
- Fixing bug Exceed maximum recursion depth bug
- Fixing docs
- Updating README

[​](https://docs.crewai.com/en/changelog#mar-04%2C-2024)

Mar 04, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-19-0)  v0.19.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.19.0)

- Efficiency in tool usage +1023.21%
- Mean tools used +276%
- Tool errors slashed by 67%, more reliable than ever.
- Delegation capabilities enhanced
- Ability to fallback to function calling by setting `function_calling_llm` to Agent or Crew
- Ability to get crew execution metrics after `kickoff` with `crew.usage_metrics`
- Adding ability for inputs being passed in kickoff now `crew.kickoff(inputs: {'key': 'value})`
- Updating Docs

[​](https://docs.crewai.com/en/changelog#feb-28%2C-2024)

Feb 28, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-16-3)  v0.16.3

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.16.3)

- Fixing overall bugs
- Making sure code is backwards compatible

[​](https://docs.crewai.com/en/changelog#feb-28%2C-2024-2)

Feb 28, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-16-0)  v0.16.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.16.0)

- Removing lingering `crewai_tools` dependency
- Adding initial support for inputs interpolation (missing docs)
- Adding ability to track tools usage, tools error, formatting errors, tokens usage
- Updating README

[​](https://docs.crewai.com/en/changelog#feb-26%2C-2024)

Feb 26, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-14-4)  v0.14.4

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.14.4)

- Updating timeouts
- Updating docs
- Removing crewai\_tools as a mandatory
- Making agents memory-less by default for token count reduction (breaking change for people counting on this previously)

[​](https://docs.crewai.com/en/changelog#feb-24%2C-2024)

Feb 24, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-14-3)  v0.14.3

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.14.3)

- Fixing broken docs link
- Adding support for agents without tools
- Avoid empty task outputs

[​](https://docs.crewai.com/en/changelog#feb-22%2C-2024)

Feb 22, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-14-0)  v0.14.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.14.0)All improvements from the v0.14.0rc.

- Support to export json and pydantic from opensource models

[​](https://docs.crewai.com/en/changelog#feb-20%2C-2024)

Feb 20, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-14-0rc)  v0.14.0rc

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.14.0rc0)

- Adding support to crewai-tools
- Adding support to format tasks output as Pydantic Objects Or JSON
- Adding support to save tasks ouput to a file
- Improved reliability for inter agent delegation
- Revamp tools usage logic to proper use function calling
- Updating internal prompts
- Supporting tools with no arguments
- Bug fixes

[​](https://docs.crewai.com/en/changelog#feb-16%2C-2024)

Feb 16, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-11-2)  v0.11.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.11.2)

- Adding further error logging so users understand what is happening if a tool fails

[​](https://docs.crewai.com/en/changelog#feb-16%2C-2024-2)

Feb 16, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-11-1)  v0.11.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.11.1)

- It fixes a bug on the tool usage logic that was early caching the result even if there was an error on the usage, preventing it from using the tool again.
- It will also print any error message in red allowing the user to understand what was the problem with the tool.

[​](https://docs.crewai.com/en/changelog#feb-13%2C-2024)

Feb 13, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-11-0)  v0.11.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.11.0)

- Ability to set `function_calling_llm` on both the entire crew and individual agents
- Some early attempts on cost reduction
- Improving function calling for tools
- Updates docs

[​](https://docs.crewai.com/en/changelog#feb-10%2C-2024)

Feb 10, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-10-0)  v0.10.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.10.0)

- Ability to get `full_ouput` from crew kickoff with all tasks outputs
- Ability to set `step_callback` function for both Agents and Crews so you can get all intermediate steps
- Remembering Agent of the expected format after certain number of tool usages.
- New tool usage internals now using json, unlocking tools with multiple arguments
- Refactoring overall delegation logic, now way more reliable
- Fixed `max_inter` bug now properly forcing llm to answer as it gets to that
- Rebuilt caching structure, making sure multiple agents can use the same cache
- Refactoring Task repeated usage prevention logic
- Removing now unnecessary `CrewAgentOutputParser`
- Opt-in to share complete crew related data with the crewAI team
- Overall Docs update

[​](https://docs.crewai.com/en/changelog#feb-08%2C-2024)

Feb 08, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-5-5)  v0.5.5

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.5.5)

- Overall doc + readme improvements
- Fixing RPM controller being set unnecessarily
- Adding early stage anonymous telemetry for lib improvement

[​](https://docs.crewai.com/en/changelog#feb-07%2C-2024)

Feb 07, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-5-3)  v0.5.3

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.5.3)

- quick Fix for hierarchical manager

[​](https://docs.crewai.com/en/changelog#feb-06%2C-2024)

Feb 06, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-5-2)  v0.5.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.5.2)

- Adding `manager_llm` for hierarchical process
- Improving `max_inter` and `max_rpm` logic
- Updating README and Docs

[​](https://docs.crewai.com/en/changelog#feb-04%2C-2024)

Feb 04, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-5-0)  v0.5.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.5.0)This new version bring a lot of new features and improvements to the library.

## [​](https://docs.crewai.com/en/changelog\#features-2)  Features

- Adding Task Callbacks.
- Adding support for Hierarchical process.
- Adding ability to references specific tasks in another task.
- Adding ability to parallel task execution.

## [​](https://docs.crewai.com/en/changelog\#improvements)  Improvements

- Revamping Max Iterations and Max Requests per Minute.
- Developer experience improvements, docstrings and such.
- Small improvements and TYPOs.
- Fix static typing errors.
- Updated README and Docs.

[​](https://docs.crewai.com/en/changelog#jan-14%2C-2024)

Jan 14, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-1-32)  v0.1.32

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.1.32)

- Moving to LangChain 0.1.0
- Improving Prompts
- Adding ability to limit maximum number of iterations for an agent
- Adding ability to Request Per Minute throttling for both Agents and Crews
- Adding initial support for translations
- Adding Greek translation
- Improve code readability
- Starting new documentation with mkdocs

[​](https://docs.crewai.com/en/changelog#jan-07%2C-2024)

Jan 07, 2024

## [​](https://docs.crewai.com/en/changelog\#v0-1-23)  v0.1.23

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.1.23)

- Many Reliability improvements
- Prompt changes
- Initial changes for supporting multiple languages
- Fixing bug on task repeated execution
- Better execution error handling
- Updating READMe

[​](https://docs.crewai.com/en/changelog#dec-30%2C-2023)

Dec 30, 2023

## [​](https://docs.crewai.com/en/changelog\#v0-1-14)  v0.1.14

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.1.14)

- Adding tool caching a loop execution prevention. (@joaomdmoura)
- Adding more guidelines for Agent delegation. (@joaomdmoura)
- Updating to use new openai lib version. (@joaomdmoura)
- Adding verbose levels to the logger. (@joaomdmoura)
- Removing WIP code. (@joaomdmoura)
- A lot of developer quality of life improvements (Special thanks to @greysonlalonde).
- Updating to pydantic v2 (Special thanks to @greysonlalonde as well).

[​](https://docs.crewai.com/en/changelog#nov-24%2C-2023)

Nov 24, 2023

## [​](https://docs.crewai.com/en/changelog\#v0-1-2)  v0.1.2

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.1.2)

- Adding ability to use other LLMs, not OpenAI

[​](https://docs.crewai.com/en/changelog#nov-19%2C-2023)

Nov 19, 2023

## [​](https://docs.crewai.com/en/changelog\#v0-1-1)  v0.1.1

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.1.1)

# [​](https://docs.crewai.com/en/changelog\#crewai-v0-1-1-release-notes)  CrewAI v0.1.1 Release Notes

## [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-new)  What’s New

- **Crew Verbose Mode**: Now allowing you to inspect a the tasks are being executed.
- **README and Docs Updates**: A series of minor updates on the docs

[​](https://docs.crewai.com/en/changelog#nov-14%2C-2023)

Nov 14, 2023

## [​](https://docs.crewai.com/en/changelog\#v0-1-0)  v0.1.0

[View release on GitHub](https://github.com/crewAIInc/crewAI/releases/tag/v0.1.0)

# [​](https://docs.crewai.com/en/changelog\#crewai-v0-1-0-release-notes)  CrewAI v0.1.0 Release Notes

We are thrilled to announce the initial release of CrewAI, version 0.1.0! CrewAI is a framework designed to facilitate the orchestration of autonomous AI agents capable of role-playing and collaboration to accomplish complex tasks more efficiently.

## [​](https://docs.crewai.com/en/changelog\#what%E2%80%99s-new-2)  What’s New

- **Initial Launch**: CrewAI is now officially in the wild! This foundational release lays the groundwork for AI agents to work in tandem, each with its own specialized role and objectives.
- **Role-Based Agent Design**: Define and customize agents with specific roles, goals, and the tools they need to succeed.
- **Inter-Agent Delegation**: Agents are now equipped to autonomously delegate tasks, enabling dynamic distribution of workload among the team.
- **Task Management**: Create and assign tasks dynamically with the flexibility to specify the tools needed for each task.
- **Sequential Processes**: Set up your agents to tackle tasks one after the other, ensuring organized and predictable workflows.
- **Documentation**: Start exploring CrewAI with our initial documentation that guides you through the setup and use of the framework.

## [​](https://docs.crewai.com/en/changelog\#enhancements)  Enhancements

- Detailed API documentation for the `Agent`, `Task`, `Crew`, and `Process` classes.
- Examples and tutorials to help you build your first CrewAI application.
- Basic setup for collaborative and delegation mechanisms among agents.

## [​](https://docs.crewai.com/en/changelog\#known-issues)  Known Issues

- As this is the first release, there may be undiscovered bugs and areas for optimization. We encourage the community to report any issues found during use.

## [​](https://docs.crewai.com/en/changelog\#upcoming-features)  Upcoming Features

- **Advanced Process Management**: In future releases, we will introduce more complex processes for task management including consensual and hierarchical workflows.

Was this page helpful?

YesNo

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.
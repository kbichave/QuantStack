[Skip to main content](https://docs.crewai.com/en/api-reference/introduction#content-area)

[CrewAI home page![light logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)![dark logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)](https://docs.crewai.com/)

![US](https://d3gk2c5xim1je2.cloudfront.net/flags/US.svg)

English

Search...

Ctrl K

Search...

Navigation

Getting Started

Introduction

[Home](https://docs.crewai.com/) [Documentation](https://docs.crewai.com/en/introduction) [AOP](https://docs.crewai.com/en/enterprise/introduction) [API Reference](https://docs.crewai.com/en/api-reference/introduction) [Examples](https://docs.crewai.com/en/examples/example) [Changelog](https://docs.crewai.com/en/changelog)

- [Website](https://crewai.com/)
- [Forum](https://community.crewai.com/)
- [Blog](https://blog.crewai.com/)
- [CrewGPT](https://chatgpt.com/g/g-qqTuUWsBY-crewai-assistant)

##### Getting Started

- [Introduction](https://docs.crewai.com/en/api-reference/introduction)
- [GET\\
\\
GET /inputs](https://docs.crewai.com/en/api-reference/inputs)
- [POST\\
\\
POST /kickoff](https://docs.crewai.com/en/api-reference/kickoff)
- [POST\\
\\
POST /resume](https://docs.crewai.com/en/api-reference/resume)
- [GET\\
\\
GET /status/{kickoff\_id}](https://docs.crewai.com/en/api-reference/status)

# [​](https://docs.crewai.com/en/api-reference/introduction\#crewai-aop-api)  CrewAI AOP API

Welcome to the CrewAI AOP API reference. This API allows you to programmatically interact with your deployed crews, enabling integration with your applications, workflows, and services.

## [​](https://docs.crewai.com/en/api-reference/introduction\#quick-start)  Quick Start

1

Get Your API Credentials

Navigate to your crew’s detail page in the CrewAI AOP dashboard and copy your Bearer Token from the Status tab.

2

Discover Required Inputs

Use the `GET /inputs` endpoint to see what parameters your crew expects.

3

Start a Crew Execution

Call `POST /kickoff` with your inputs to start the crew execution and receive a `kickoff_id`.

4

Monitor Progress

Use `GET /status/{kickoff_id}` to check execution status and retrieve results.

## [​](https://docs.crewai.com/en/api-reference/introduction\#authentication)  Authentication

All API requests require authentication using a Bearer token. Include your token in the `Authorization` header:

Copy

Ask AI

```
curl -H "Authorization: Bearer YOUR_CREW_TOKEN" \
  https://your-crew-url.crewai.com/inputs
```

### [​](https://docs.crewai.com/en/api-reference/introduction\#token-types)  Token Types

| Token Type | Scope | Use Case |
| --- | --- | --- |
| **Bearer Token** | Organization-level access | Full crew operations, ideal for server-to-server integration |
| **User Bearer Token** | User-scoped access | Limited permissions, suitable for user-specific operations |

You can find both token types in the Status tab of your crew’s detail page in the CrewAI AOP dashboard.

## [​](https://docs.crewai.com/en/api-reference/introduction\#base-url)  Base URL

Each deployed crew has its own unique API endpoint:

Copy

Ask AI

```
https://your-crew-name.crewai.com
```

Replace `your-crew-name` with your actual crew’s URL from the dashboard.

## [​](https://docs.crewai.com/en/api-reference/introduction\#typical-workflow)  Typical Workflow

1. **Discovery**: Call `GET /inputs` to understand what your crew needs
2. **Execution**: Submit inputs via `POST /kickoff` to start processing
3. **Monitoring**: Poll `GET /status/{kickoff_id}` until completion
4. **Results**: Extract the final output from the completed response

## [​](https://docs.crewai.com/en/api-reference/introduction\#error-handling)  Error Handling

The API uses standard HTTP status codes:

| Code | Meaning |
| --- | --- |
| `200` | Success |
| `400` | Bad Request - Invalid input format |
| `401` | Unauthorized - Invalid bearer token |
| `404` | Not Found - Resource doesn’t exist |
| `422` | Validation Error - Missing required inputs |
| `500` | Server Error - Contact support |

## [​](https://docs.crewai.com/en/api-reference/introduction\#interactive-testing)  Interactive Testing

**Why no “Send” button?** Since each CrewAI AOP user has their own unique crew URL, we use **reference mode** instead of an interactive playground to avoid confusion. This shows you exactly what the requests should look like without non-functional send buttons.

Each endpoint page shows you:

- ✅ **Exact request format** with all parameters
- ✅ **Response examples** for success and error cases
- ✅ **Code samples** in multiple languages (cURL, Python, JavaScript, etc.)
- ✅ **Authentication examples** with proper Bearer token format

### [​](https://docs.crewai.com/en/api-reference/introduction\#to-test-your-actual-api:)  **To Test Your Actual API:**

## Copy cURL Examples

Copy the cURL examples and replace the URL + token with your real values

## Use Postman/Insomnia

Import the examples into your preferred API testing tool

**Example workflow:**

1. **Copy this cURL example** from any endpoint page
2. **Replace `your-actual-crew-name.crewai.com`** with your real crew URL
3. **Replace the Bearer token** with your real token from the dashboard
4. **Run the request** in your terminal or API client

## [​](https://docs.crewai.com/en/api-reference/introduction\#need-help)  Need Help?

[**Enterprise Support** \\
\\
Get help with API integration and troubleshooting](mailto:support@crewai.com) [**Enterprise Dashboard** \\
\\
Manage your crews and view execution logs](https://app.crewai.com/)

Was this page helpful?

YesNo

[GET /inputs\\
\\
Next](https://docs.crewai.com/en/api-reference/inputs)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.
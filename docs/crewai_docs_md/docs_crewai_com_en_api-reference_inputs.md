[Skip to main content](https://docs.crewai.com/en/api-reference/inputs#content-area)

[CrewAI home page![light logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)![dark logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)](https://docs.crewai.com/)

![US](https://d3gk2c5xim1je2.cloudfront.net/flags/US.svg)

English

Search...

Ctrl K

Search...

Navigation

Getting Started

GET /inputs

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

GET

https://your-actual-crew-name.crewai.comhttps://my-travel-crew.crewai.comhttps://content-creation-crew.crewai.comhttps://research-assistant-crew.crewai.com

/

inputs

Try it

Get Required Inputs

cURL

Copy

Ask AI

```
curl --request GET \
  --url https://your-actual-crew-name.crewai.com/inputs \
  --header 'Authorization: Bearer <token>'
```

200

travel\_crew

Copy

Ask AI

```
{
  "inputs": [\
    "budget",\
    "interests",\
    "duration",\
    "age"\
  ]
}
```

#### Authorizations

[​](https://docs.crewai.com/en/api-reference/inputs#authorization-authorization)

Authorization

string

header

required

📋 Reference Documentation \- _The tokens shown in examples are placeholders for reference only._

Use your actual Bearer Token or User Bearer Token from the CrewAI AOP dashboard for real API calls.

Bearer Token: Organization-level access for full crew operations
User Bearer Token: User-scoped access with limited permissions

#### Response

200

application/json

Successfully retrieved required inputs

[​](https://docs.crewai.com/en/api-reference/inputs#response-inputs)

inputs

string\[\]

Array of required input parameter names

Example:

```
["budget", "interests", "duration", "age"]
```

Was this page helpful?

YesNo

[Introduction\\
\\
Previous](https://docs.crewai.com/en/api-reference/introduction) [POST /kickoff\\
\\
Next](https://docs.crewai.com/en/api-reference/kickoff)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.
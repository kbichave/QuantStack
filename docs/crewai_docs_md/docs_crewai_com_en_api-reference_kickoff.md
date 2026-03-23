[Skip to main content](https://docs.crewai.com/en/api-reference/kickoff#content-area)

[CrewAI home page![light logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)![dark logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)](https://docs.crewai.com/)

![US](https://d3gk2c5xim1je2.cloudfront.net/flags/US.svg)

English

Search...

Ctrl K

Search...

Navigation

Getting Started

POST /kickoff

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

POST

https://your-actual-crew-name.crewai.comhttps://my-travel-crew.crewai.comhttps://content-creation-crew.crewai.comhttps://research-assistant-crew.crewai.com

/

kickoff

Try it

cURL

travel\_planning

Copy

Ask AI

```
curl --request POST \
  --url https://your-actual-crew-name.crewai.com/kickoff \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '
{
  "inputs": {
    "budget": "1000 USD",
    "interests": "games, tech, ai, relaxing hikes, amazing food",
    "duration": "7 days",
    "age": "35"
  },
  "meta": {
    "requestId": "travel-req-123",
    "source": "web-app"
  }
}
'
```

200

400

401

422

500

Copy

Ask AI

```
{
  "kickoff_id": "abcd1234-5678-90ef-ghij-klmnopqrstuv"
}
```

#### Authorizations

[​](https://docs.crewai.com/en/api-reference/kickoff#authorization-authorization)

Authorization

string

header

required

📋 Reference Documentation \- _The tokens shown in examples are placeholders for reference only._

Use your actual Bearer Token or User Bearer Token from the CrewAI AOP dashboard for real API calls.

Bearer Token: Organization-level access for full crew operations
User Bearer Token: User-scoped access with limited permissions

#### Body

application/json

[​](https://docs.crewai.com/en/api-reference/kickoff#body-inputs)

inputs

object

required

Key-value pairs of all required inputs for your crew

Showchild attributes

[​](https://docs.crewai.com/en/api-reference/kickoff#body-inputs-additional-properties)

inputs.{key}

string

Example:

```
{
  "budget": "1000 USD",
  "interests": "games, tech, ai, relaxing hikes, amazing food",
  "duration": "7 days",
  "age": "35"
}
```

[​](https://docs.crewai.com/en/api-reference/kickoff#body-meta)

meta

object

Additional metadata to pass to the crew

Example:

```
{
  "requestId": "user-request-12345",
  "source": "mobile-app"
}
```

[​](https://docs.crewai.com/en/api-reference/kickoff#body-task-webhook-url)

taskWebhookUrl

string<uri>

Callback URL executed after each task completion

Example:

`"https://your-server.com/webhooks/task"`

[​](https://docs.crewai.com/en/api-reference/kickoff#body-step-webhook-url)

stepWebhookUrl

string<uri>

Callback URL executed after each agent thought/action

Example:

`"https://your-server.com/webhooks/step"`

[​](https://docs.crewai.com/en/api-reference/kickoff#body-crew-webhook-url)

crewWebhookUrl

string<uri>

Callback URL executed when the crew execution completes

Example:

`"https://your-server.com/webhooks/crew"`

#### Response

200

application/json

Crew execution started successfully

[​](https://docs.crewai.com/en/api-reference/kickoff#response-kickoff-id)

kickoff\_id

string<uuid>

Unique identifier for tracking this execution

Example:

`"abcd1234-5678-90ef-ghij-klmnopqrstuv"`

Was this page helpful?

YesNo

[GET /inputs\\
\\
Previous](https://docs.crewai.com/en/api-reference/inputs) [POST /resume\\
\\
Next](https://docs.crewai.com/en/api-reference/resume)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.
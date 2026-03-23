[Skip to main content](https://docs.crewai.com/en/api-reference/resume#content-area)

[CrewAI home page![light logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)![dark logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)](https://docs.crewai.com/)

![US](https://d3gk2c5xim1je2.cloudfront.net/flags/US.svg)

English

Search...

Ctrl K

Search...

Navigation

Getting Started

POST /resume

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

resume

Try it

cURL

approve\_and\_continue

Copy

Ask AI

```
curl --request POST \
  --url https://your-actual-crew-name.crewai.com/resume \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '
{
  "execution_id": "abcd1234-5678-90ef-ghij-klmnopqrstuv",
  "task_id": "research_task",
  "human_feedback": "Excellent research! Proceed to the next task.",
  "is_approve": true,
  "taskWebhookUrl": "https://api.example.com/webhooks/task",
  "stepWebhookUrl": "https://api.example.com/webhooks/step",
  "crewWebhookUrl": "https://api.example.com/webhooks/crew"
}
'
```

200

resumed

Copy

Ask AI

```
{
  "status": "resumed",
  "message": "Execution resumed successfully"
}
```

#### Authorizations

[​](https://docs.crewai.com/en/api-reference/resume#authorization-authorization)

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

[​](https://docs.crewai.com/en/api-reference/resume#body-execution-id)

execution\_id

string<uuid>

required

The unique identifier for the crew execution (from kickoff)

Example:

`"abcd1234-5678-90ef-ghij-klmnopqrstuv"`

[​](https://docs.crewai.com/en/api-reference/resume#body-task-id)

task\_id

string

required

The ID of the task that requires human feedback

Example:

`"research_task"`

[​](https://docs.crewai.com/en/api-reference/resume#body-human-feedback)

human\_feedback

string

required

Your feedback on the task output. This will be incorporated as additional context for subsequent task executions.

Example:

`"Great research! Please add more details about recent developments in the field."`

[​](https://docs.crewai.com/en/api-reference/resume#body-is-approve)

is\_approve

boolean

required

Whether you approve the task output: true = positive feedback (continue), false = negative feedback (retry task)

Example:

`true`

[​](https://docs.crewai.com/en/api-reference/resume#body-task-webhook-url)

taskWebhookUrl

string<uri>

Callback URL executed after each task completion. MUST be provided to continue receiving task notifications.

Example:

`"https://your-server.com/webhooks/task"`

[​](https://docs.crewai.com/en/api-reference/resume#body-step-webhook-url)

stepWebhookUrl

string<uri>

Callback URL executed after each agent thought/action. MUST be provided to continue receiving step notifications.

Example:

`"https://your-server.com/webhooks/step"`

[​](https://docs.crewai.com/en/api-reference/resume#body-crew-webhook-url)

crewWebhookUrl

string<uri>

Callback URL executed when the crew execution completes. MUST be provided to receive completion notification.

Example:

`"https://your-server.com/webhooks/crew"`

#### Response

200

application/json

Execution resumed successfully

[​](https://docs.crewai.com/en/api-reference/resume#response-status)

status

enum<string>

Status of the resumed execution

Available options:

`resumed`,

`retrying`,

`completed`

Example:

`"resumed"`

[​](https://docs.crewai.com/en/api-reference/resume#response-message)

message

string

Human-readable message about the resume operation

Example:

`"Execution resumed successfully"`

Was this page helpful?

YesNo

[POST /kickoff\\
\\
Previous](https://docs.crewai.com/en/api-reference/kickoff) [GET /status/{kickoff\_id}\\
\\
Next](https://docs.crewai.com/en/api-reference/status)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.
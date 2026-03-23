[Skip to main content](https://docs.crewai.com/en/api-reference/status#content-area)

[CrewAI home page![light logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)![dark logo](https://mintcdn.com/crewai/5SZbe87tsCWZY09V/images/crew_only_logo.png?fit=max&auto=format&n=5SZbe87tsCWZY09V&q=85&s=439ca5dc63a1768cad7196005ff5636f)](https://docs.crewai.com/)

![US](https://d3gk2c5xim1je2.cloudfront.net/flags/US.svg)

English

Search...

Ctrl K

Search...

Navigation

Getting Started

GET /status/{kickoff\_id}

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

status

/

{kickoff\_id}

Try it

Get Execution Status

cURL

Copy

Ask AI

```
curl --request GET \
  --url https://your-actual-crew-name.crewai.com/status/{kickoff_id} \
  --header 'Authorization: Bearer <token>'
```

200

running

Copy

Ask AI

```
{
  "status": "running",
  "current_task": "research_task",
  "progress": {
    "completed_tasks": 1,
    "total_tasks": 3
  }
}
```

#### Authorizations

[​](https://docs.crewai.com/en/api-reference/status#authorization-authorization)

Authorization

string

header

required

📋 Reference Documentation \- _The tokens shown in examples are placeholders for reference only._

Use your actual Bearer Token or User Bearer Token from the CrewAI AOP dashboard for real API calls.

Bearer Token: Organization-level access for full crew operations
User Bearer Token: User-scoped access with limited permissions

#### Path Parameters

[​](https://docs.crewai.com/en/api-reference/status#parameter-kickoff-id)

kickoff\_id

string<uuid>

required

The kickoff ID returned from the /kickoff endpoint

Example:

`"abcd1234-5678-90ef-ghij-klmnopqrstuv"`

#### Response

200

application/json

Successfully retrieved execution status

- Option 1

- Option 2

- Option 3


[​](https://docs.crewai.com/en/api-reference/status#response-one-of-0-status)

status

enum<string>

Available options:

`running`

Example:

`"running"`

[​](https://docs.crewai.com/en/api-reference/status#response-one-of-0-current-task)

current\_task

string

Name of the currently executing task

Example:

`"research_task"`

[​](https://docs.crewai.com/en/api-reference/status#response-one-of-0-progress)

progress

object

Showchild attributes

[​](https://docs.crewai.com/en/api-reference/status#response-one-of-0-progress-completed-tasks)

progress.completed\_tasks

integer

Number of completed tasks

Example:

`1`

[​](https://docs.crewai.com/en/api-reference/status#response-one-of-0-progress-total-tasks)

progress.total\_tasks

integer

Total number of tasks in the crew

Example:

`3`

Was this page helpful?

YesNo

[POST /resume\\
\\
Previous](https://docs.crewai.com/en/api-reference/resume)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.
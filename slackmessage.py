import os
from slack import WebClient
from slack.errors import SlackApiError


def send_slack_message(filepath):
    client = WebClient(token=os.environ['SLACK_API_TOKEN'])

    try:
        response = client.files_upload(
            channels='#detections',
            file=filepath)
        assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")
    
    os._exit(0) 
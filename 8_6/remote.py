from __future__ import print_function
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaUpload, MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and not creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive','v3', credentials=creds)


def upload_file(service, local_file_path, drive_folder_id):
    file_metadata = {
        'name': os.path.basename(local_file_path),
        'parents': [drive_folder_id]
    }
    media = MediaFileUpload(local_file_path, resumable=True)
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    # print(f'Uploaded file "{local_file_path}" with file ID: {file.get({"id"})}')


def upload_directory(local_directory, drive_folder_id):
    service = authenticate()
    for filename in os.listdir(local_directory):
        file_path = os.path.join(local_directory, filename)
        if os.path.isfile(file_path):
            upload_file(service, file_path, drive_folder_id)


def upload(local):
    drive_folder_id = "1oTyx-IHwt3YH0CtU0IcMhEWQtTkCZ2Eg"
    upload_directory(local, drive_folder_id)

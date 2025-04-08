"""
Handles Fyers API authentication, token management, and WebSocket setup.
Initializes the fyers client instance for use across the project.
"""

import base64
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import pyotp
import pytz
import requests
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

# Credentials (replace with secure loading in production)
app_id = "TS79V3NXK1-100"
secret_key = "KQCPB0FJ74"
redirect_uri = "https://google.com"
fyers_user = "XM22383"
fyers_pin = "4628"
fyers_totp = "EAQD6K4IUYOEGPJNVE6BMPTUSDCWIOHW"
response_type = "code"
grant_type = "authorization_code"

ist_timezone = pytz.timezone("Asia/Kolkata")

def get_encoded_string(string):
    string = str(string)
    base64_bytes = base64.b64encode(string.encode("ascii"))
    return base64_bytes.decode("ascii")

def authenticate():
    session = fyersModel.SessionModel(
        client_id=app_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type=response_type,
        grant_type=grant_type,
    )

    session.generate_authcode()

    url_send_login_otp = "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
    res = requests.post(
        url=url_send_login_otp,
        json={"fy_id": get_encoded_string(fyers_user), "app_id": "2"},
    ).json()

    if datetime.now().second % 30 > 27:
        time.sleep(5)

    url_verify_otp = "https://api-t2.fyers.in/vagator/v2/verify_otp"
    res2 = requests.post(
        url=url_verify_otp,
        json={
            "request_key": res["request_key"],
            "otp": pyotp.TOTP(fyers_totp).now(),
        },
    ).json()

    ses = requests.Session()
    url_verify_otp2 = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
    payload2 = {
        "request_key": res2["request_key"],
        "identity_type": "pin",
        "identifier": get_encoded_string(fyers_pin),
    }
    res3 = ses.post(url=url_verify_otp2, json=payload2).json()

    ses.headers.update({"authorization": f"Bearer {res3['data']['access_token']}"})

    tokenurl = "https://api-t1.fyers.in/api/v3/token"
    payload3 = {
        "fyers_id": fyers_user,
        "app_id": app_id[:-4],
        "redirect_uri": redirect_uri,
        "appType": "100",
        "code_challenge": "",
        "state": "None",
        "scope": "",
        "nonce": "",
        "response_type": "code",
        "create_cookie": True,
    }

    res3 = ses.post(url=tokenurl, json=payload3).json()

    url = res3["Url"]
    parsed = urlparse(url)
    auth_code = parse_qs(parsed.query)["auth_code"][0]

    session.set_token(auth_code)

    auth_response = session.generate_token()
    access_token = auth_response["access_token"]

    fyers_client = fyersModel.FyersModel(client_id=app_id, token=access_token)

    return fyers_client

# Initialize fyers client at module load
try:
    fyers = authenticate()
except Exception as e:
    print(f"Error authenticating with Fyers: {e}")
    fyers = None

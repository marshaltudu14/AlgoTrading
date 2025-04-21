import base64
import requests
import pyotp
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import pytz
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

ist_timezone = pytz.timezone("Asia/Kolkata")

def get_encoded_string(s):
    return base64.b64encode(str(s).encode("ascii")).decode("ascii")


def authenticate_fyers(app_id, secret_key, redirect_uri, fyers_user, fyers_pin, fyers_totp,
                       response_type="code", grant_type="authorization_code"):  
    session = fyersModel.SessionModel(
        client_id=app_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type=response_type,
        grant_type=grant_type
    )
    session.generate_authcode()
    # send login OTP
    url_send_login_otp = "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
    res = requests.post(url=url_send_login_otp,
                        json={"fy_id": get_encoded_string(fyers_user), "app_id": "2"}).json()
    # wait if needed
    if datetime.now(ist_timezone).second % 30 > 27:
        import time; time.sleep(5)
    # verify OTP
    url_verify_otp = "https://api-t2.fyers.in/vagator/v2/verify_otp"
    res2 = requests.post(url=url_verify_otp,
                         json={"request_key": res.get("request_key"),
                               "otp": pyotp.TOTP(fyers_totp).now()}).json()
    # verify PIN
    ses = requests.Session()
    url_verify_pin = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
    payload2 = {"request_key": res2.get("request_key"),
                "identity_type": "pin",
                "identifier": get_encoded_string(fyers_pin)}
    res3 = ses.post(url=url_verify_pin, json=payload2).json()
    ses.headers.update({'authorization': f"Bearer {res3['data']['access_token']}"})
    # get auth code
    tokenurl = "https://api-t1.fyers.in/api/v3/token"
    payload3 = {
        "fyers_id": fyers_user,
        "app_id": app_id[:-4],
        "redirect_uri": redirect_uri,
        "appType": app_id.split('-')[-1],
        "code_challenge": "",
        "state": "None",
        "scope": "",
        "nonce": "",
        "response_type": response_type,
        "create_cookie": True
    }
    res4 = ses.post(url=tokenurl, json=payload3).json()
    url = res4.get('Url', '')
    parsed = urlparse(url)
    auth_code = parse_qs(parsed.query).get('auth_code', [''])[0]
    session.set_token(auth_code)
    auth_response = session.generate_token()
    access_token = auth_response.get("access_token")
    fyers = fyersModel.FyersModel(client_id=app_id, token=access_token)
    ws_token = f"{app_id}:{access_token}"
    fyers_socket = data_ws.FyersDataSocket(access_token=ws_token, log_path="")
    return fyers, fyers_socket

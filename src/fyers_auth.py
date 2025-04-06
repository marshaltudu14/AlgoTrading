import requests
import base64
import pyotp
import time
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from fyers_apiv3 import fyersModel

# Import config variables - assumes this script is run where 'src' is importable
# Or adjust import path if run differently
try:
    from . import config
except ImportError:
    import config # Fallback for running script directly


def get_encoded_string(string):
    """Encodes a string to Base64."""
    string = str(string)
    base64_bytes = base64.b64encode(string.encode("ascii"))
    return base64_bytes.decode("ascii")

def get_fyers_access_token():
    """Handles the Fyers authentication flow and returns the access token."""
    try:
        session = fyersModel.SessionModel(
            client_id=config.APP_ID,
            secret_key=config.SECRET_KEY,
            redirect_uri=config.REDIRECT_URI,
            response_type=config.RESPONSE_TYPE,
            grant_type=config.GRANT_TYPE
        )

        # Step 1: Generate Auth Code URL (though not directly used here)
        # session.generate_authcode() # This might open a browser if not handled

        # Step 2: Send Login OTP
        url_send_login_otp = "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
        otp_payload = {"fy_id": get_encoded_string(config.FYERS_USER), "app_id": "2"}
        otp_res = requests.post(url=url_send_login_otp, json=otp_payload).json()
        print(f"OTP Sent Response: {otp_res}")
        if otp_res.get("status") != "success" and otp_res.get("s") != "ok":
             raise Exception(f"Failed to send OTP: {otp_res.get('message', 'Unknown error')}")


        # Handle potential delay needed for TOTP generation
        if datetime.now().second % 30 > 27:
            print("Waiting briefly for TOTP window...")
            time.sleep(5)

        # Step 3: Verify OTP (using TOTP)
        url_verify_otp = "https://api-t2.fyers.in/vagator/v2/verify_otp"
        totp_code = pyotp.TOTP(config.FYERS_TOTP_KEY).now()
        verify_otp_payload = {"request_key": otp_res["request_key"], "otp": totp_code}
        verify_otp_res = requests.post(url=url_verify_otp, json=verify_otp_payload).json()
        print(f"Verify OTP Response: {verify_otp_res}")
        if verify_otp_res.get("status") != "success" and verify_otp_res.get("s") != "ok":
             raise Exception(f"Failed to verify TOTP: {verify_otp_res.get('message', 'Unknown error')}")


        # Step 4: Verify PIN
        ses = requests.Session() # Use a session for subsequent requests
        url_verify_pin = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
        pin_payload = {
            "request_key": verify_otp_res["request_key"],
            "identity_type": "pin",
            "identifier": get_encoded_string(config.FYERS_PIN)
        }
        verify_pin_res = ses.post(url=url_verify_pin, json=pin_payload).json()
        print(f"Verify PIN Response: {verify_pin_res}")
        if verify_pin_res.get("s") != "ok":
             raise Exception(f"Failed to verify PIN: {verify_pin_res.get('message', 'Unknown error')}")


        # Step 5: Get Auth Code via Token Endpoint (using session cookies)
        ses.headers.update({'authorization': f"Bearer {verify_pin_res['data']['access_token']}"})
        token_url = "https://api-t1.fyers.in/api/v3/token"
        token_payload = {
            "fyers_id": config.FYERS_USER,
            "app_id": config.APP_ID[:-4], # App ID without '-100'
            "redirect_uri": config.REDIRECT_URI,
            "appType": "100",
            "code_challenge": "",
            "state": config.STATE,
            "scope": "",
            "nonce": "",
            "response_type": config.RESPONSE_TYPE,
            "create_cookie": True
        }
        token_res = ses.post(url=token_url, json=token_payload).json()
        print(f"Token URL Response: {token_res}")
        if not token_res.get('Url'):
             raise Exception(f"Failed to get auth code URL: {token_res.get('message', 'Unknown error')}")


        # Step 6: Extract Auth Code from Redirect URL
        redirect_url = token_res['Url']
        parsed_url = urlparse(redirect_url)
        auth_code = parse_qs(parsed_url.query).get('auth_code', [None])[0]
        if not auth_code:
            raise Exception("Could not extract auth_code from URL.")
        print(f"Extracted Auth Code: {auth_code[:5]}...") # Print only prefix for security


        # Step 7: Generate Final Access Token
        session.set_token(auth_code)
        auth_response = session.generate_token()
        print(f"Generate Token Response: {auth_response}")
        if not auth_response.get("access_token"):
             raise Exception(f"Failed to generate final access token: {auth_response.get('message', 'Unknown error')}")

        access_token = auth_response["access_token"]
        formatted_token = f"{config.APP_ID}:{access_token}" # Format for WebSocket
        print("Successfully obtained Fyers Access Token.")
        # Return both raw and formatted tokens
        return {
            "access_token": access_token, # Raw token for fyersModel (REST)
            "ws_token": formatted_token   # Formatted token for WebSockets
        }

    except requests.exceptions.RequestException as e:
        print(f"Network error during Fyers authentication: {e}")
        raise
    except KeyError as e:
        print(f"Missing key in Fyers API response: {e}")
        raise
    except Exception as e:
        print(f"An error occurred during Fyers authentication: {e}")
        raise

if __name__ == "__main__":
    # Example usage when running the script directly
    try:
        token_info = get_fyers_access_token()
        if token_info:
            print(f"\nSuccessfully retrieved Access Token (prefix): {token_info['access_token'][:10]}...")
            print(f"WebSocket Token (prefix): {token_info['ws_token'][:15]}...")
            # Initialize FyersModel here if needed for testing
            # fyers = fyersModel.FyersModel(client_id=config.APP_ID, token=token_info['access_token'])
            # profile = fyers.get_profile()
            # print("\nFyers Profile:", profile)
        else:
            print("\nFailed to retrieve access token.")
    except Exception as e:
        print(f"\nAuthentication failed: {e}")

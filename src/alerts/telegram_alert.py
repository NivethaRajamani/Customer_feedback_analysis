import os 
import requests
from dotenv import load_dotenv

load_dotenv()


TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

API_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

def send_negative_alert(review):

    message = (
        "NEGATIVE REVIEW ALERT\n\n"
        f"{review}"
    )

    payload = {
        "chat_id" : CHAT_ID,
        "text" : message
    }

    response = requests.post(
        API_URL,
        data = payload,
        timeout = 10 
    )

    print("Telegram status code:", response.status_code)
    print("Telegram response:", response.text)

    response.raise_for_status
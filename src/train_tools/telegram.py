import os
import requests
import datetime


class TelegramSend:
    SILENT_TIME_EVENING = 23
    SILENT_TIME_MORNING = 7
    URL = "https://api.telegram.org/bot{0}/sendMessage?chat_id={1}&text={2}&disable_notification={3}"

    def __init__(self, alias, api_key=None, chat_id=None):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("TELEGRAM_TOKEN")

        if chat_id is not None:
            self.chat_id = chat_id
        else:
            self.chat_id = os.getenv("CHAT_ID")

        self.alias = alias

    def send(self, message):
        if self.chat_id is not None and self.api_key is not None:
            try:
                silent = self.check_silent()
                message = f"[{self.alias}]: {message}"
                url = self.URL.format(self.api_key, self.chat_id, message, silent)
                resp = requests.get(url).json()
            except Exception as e:
                print("Telegram send fail")
                print(e)
        else:
            print("Telegram send not initialized")

    def check_silent(self):
        now = datetime.datetime.now()
        if (now.hour > self.SILENT_TIME_EVENING) or (now.hour < self.SILENT_TIME_MORNING):
            return True
        else:
            return False
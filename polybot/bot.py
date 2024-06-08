import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3
import requests


class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):

    def __init__(self, token, telegram_chat_url):
        super().__init__(token, telegram_chat_url)
        self.s3 = boto3.client('s3')
        self.bucket_name = os.environ['BUCKET_NAME']
        self.yolo5_url = "http://yolo5:8081"

    def upload_to_s3(self, file_path , chat_id):
        file_name = os.path.basename(file_path)
        try:
            self.s3.upload_file(file_path, self.bucket_name, file_name)
            logger.info(f"Uploaded {file_name} to S3 bucket {self.bucket_name}")
            self.send_text(chat_id, "Image successfully uploaded to S3.")
            return file_name
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            return None

    def get_yolo5_prediction(self, img_name):
        url = f"{self.yolo5_url}/predict?imgName={img_name}"
        try:
            response = requests.post(url)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Yolo5 prediction failed with status code {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error requesting Yolo5 prediction: {e}")
            return None

    def format_prediction_message(self, prediction):
        if not prediction:
            return "Sorry, I couldn't detect any objects in your image."

        objects = {}
        for label in prediction['labels']:
            objects[label['class']] = objects.get(label['class'], 0) + 1

        object_list = [f"{count} {obj}" for obj, count in objects.items()]
        return "I found the following objects in your image: " + ", ".join(object_list) + "."

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)
            chat_id = msg['chat']['id']

            # Upload the photo to S3
            s3_image_name = self.upload_to_s3(photo_path , chat_id)
            if not s3_image_name:
                self.send_text(chat_id, "Sorry, I couldn't upload your image. Please try again.")
                return

            # Send an HTTP request to the `yolo5` service for prediction
            prediction = self.get_yolo5_prediction(s3_image_name)
            if not prediction:
                self.send_text(chat_id, "Sorry, I couldn't process your image. Please try again.")
                return

            # Format and send the results to the Telegram end-user
            result_message = self.format_prediction_message(prediction)
            self.send_text(chat_id, result_message)


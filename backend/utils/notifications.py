import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSystem:
    def __init__(self):
        self.alert_history = []
        self.last_sms_time = 0
        self.sms_cooldown = 60
        self.enabled = True
        self._load_config()

    def _load_config(self):
        self.email_enabled = bool(os.getenv("ALERT_EMAIL", ""))
        self.sms_enabled = bool(os.getenv("TWILIO_ACCOUNT_SID", ""))
        self.email_from = os.getenv("ALERT_EMAIL_FROM", "")
        self.email_to = os.getenv("ALERT_EMAIL_TO", "")
        self.email_password = os.getenv("ALERT_EMAIL_PASSWORD", "")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))

    def trigger_alert(self, alert_type: str, message: str, severity: str = "warning"):
        timestamp = time.time()
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": timestamp,
            "formatted_time": time.strftime("%H:%M:%S", time.localtime(timestamp)),
        }
        self.alert_history.append(alert)
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)
        logger.warning(f"[{alert['formatted_time']}] {alert_type.upper()}: {message}")
        if severity == "critical":
            threading.Thread(target=self._send_email_alert, args=(alert_type, message), daemon=True).start()

    def get_recent_alerts(self, limit: int = 20):
        return self.alert_history[-limit:]

    def _send_email_alert(self, alert_type: str, message: str):
        if not self.email_enabled:
            return
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_from
            msg["To"] = self.email_to
            msg["Subject"] = f"Driver Monitor ALERT: {alert_type}"
            body = f"""
Driver Monitoring System Alert
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Type: {alert_type}
Message: {message}

Action Required: Please check on the driver immediately.
            """
            msg.attach(MIMEText(body, "plain"))
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email_from, self.email_password)
                server.sendmail(self.email_from, self.email_to, msg.as_string())
            logger.info(f"Email alert sent: {alert_type}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def send_sms(self, message: str):
        now = time.time()
        if now - self.last_sms_time < self.sms_cooldown:
            logger.info("SMS cooldown active, skipping")
            return
        if not self.sms_enabled:
            logger.info(f"SMS not enabled. Would send: {message}")
            return
        try:
            from twilio.rest import Client
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
            emergency_number = os.getenv("EMERGENCY_CONTACTS")
            client = Client(account_sid, auth_token)
            client.messages.create(
                body=message,
                from_=twilio_number,
                to=emergency_number,
            )
            self.last_sms_time = now
            logger.info("SMS alert sent")
        except ImportError:
            logger.warning("twilio not installed — cannot send SMS")
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")


alert_system = AlertSystem()

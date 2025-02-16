from datetime import datetime
from typing import Protocol


class Command(Protocol):
    timestamp: datetime


class SendEmail:
    timestamp: datetime
    email: str


class Notification(Protocol):
    def send_notification(self, command: Command) -> None:
        ...


class EmailNotification:
    def send_notification(self, command: SendEmail) -> None:
        self._send_email(command.email)

    def _send_email(self, email: str) -> None:
        ...

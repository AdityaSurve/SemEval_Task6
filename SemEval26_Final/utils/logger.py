import sys
from colorama import init, Fore, Back, Style
import emoji

init(autoreset=True)


class Logger:
    def __init__(self):
        self.reset_style = Style.RESET_ALL

    def announce(self, text):
        print(f"{Fore.CYAN}{'='*40}")
        print(f"{Fore.CYAN}{' ' * 10}{text}")
        print(f"{Fore.CYAN}{'='*40}\n")

    def success(self, text):
        print(f"{Fore.GREEN}{emoji.emojize(':check_mark_button:')} {text}")

    def warning(self, text):
        print(f"{Fore.YELLOW}{emoji.emojize(':warning:')} {text}")

    def error(self, text):
        print(f"{Fore.RED}{emoji.emojize(':x:')} {text}")

    def plain(self, text):
        print(f"{Fore.WHITE}{text}")

    def log(self, text, message_type="plain"):
        if message_type == "announce":
            self.announce(text)
        elif message_type == "success":
            self.success(text)
        elif message_type == "warning":
            self.warning(text)
        elif message_type == "error":
            self.error(text)
        else:
            self.plain(text)

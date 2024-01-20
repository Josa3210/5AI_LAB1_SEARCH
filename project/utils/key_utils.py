from msvcrt import kbhit, getch
from os import _exit
from time import sleep 
from threading import Thread
from typing import Callable

def _detect_key(key, callback):
    while True:
        if kbhit():
            character = getch()
            if character in [b'\x1a', b'\x03']:
                _exit(0)
            if character.decode() == key:
                callback(character.decode())
        else:
            sleep(1)
            
def subToKey(key: str, callback: Callable[[str, Callable[[str], None]], None]):
    thread = Thread(target=_detect_key, args=[key, callback])
    thread.start()
import subprocess

class CameraController:
    def __init__(self):
        self.adb = "D:\\platform-tools\\adb -s emulator-5554"
        self.duration = 500

    def _swipe(self, x1, y1, x2, y2):
        subprocess.run(
            f'{self.adb} shell input swipe {x1} {y1} {x2} {y2} {self.duration}',
            shell=True
        )

    def up(self):
        self._swipe(100, 100, 100, 300)

    def down(self):
        self._swipe(100, 300, 100, 100)

    def left(self):
        self._swipe(100, 200, 300, 200)

    def right(self):
        self._swipe(300, 200, 100, 200)

"""
COMANDI PER AVVIARE IL PTZ
D:\platform-tools\adb kill-server
D:\platform-tools\adb start-server
D:\platform-tools\adb devices
"""
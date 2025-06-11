import asyncio
from bleak import BleakClient, BleakScanner

class HeartRateMonitor:
    HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

    def __init__(self, mac_address):
        self.mac_address = mac_address
        self.bpm = None
        self.client = BleakClient(mac_address)

    def parse_heart_rate(self, data):
        flag = data[0]
        hr_format = flag & 0x01
        hr = int(data[1]) if hr_format == 0 else int.from_bytes(data[1:3], byteorder="little")
        return hr

    def handle_notify(self, _, data):
        self.bpm = self.parse_heart_rate(data)

    async def connect(self):
        await self.client.connect()
        print("🔗 Connected to heart rate device.")

    async def disconnect(self):
        await self.client.disconnect()
        print("❌ Disconnected from heart rate device.")

    async def read_heart_rate_async(self, timeout=10):
        await self.connect()
        await self.client.start_notify(self.HR_CHAR_UUID, self.handle_notify)

        for _ in range(timeout):
            if self.bpm is not None:
                break
            await asyncio.sleep(1)

        await self.client.stop_notify(self.HR_CHAR_UUID)
        await self.disconnect()
        return self.bpm

    def read_heart_rate(self, timeout=10):
        return asyncio.run(self.read_heart_rate_async(timeout))

# ✅ Example Usage
if __name__ == "__main__":
    WATCH_MAC = "XX:XX:XX:XX:XX:XX"  # Replace with your watch MAC address
    monitor = HeartRateMonitor(WATCH_MAC)
    bpm = monitor.read_heart_rate()
    print(f"❤️ Heart Rate: {bpm} BPM")

import asyncio
from bleak import BleakClient, BleakScanner

# Heart Rate Service UUID and Characteristic
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

def parse_heart_rate(data):
    flag = data[0]
    hr_format = flag & 0x01
    hr = int(data[1]) if hr_format == 0 else int.from_bytes(data[1:3], byteorder="little")
    return hr

async def run():
    print("Scanning...")
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)  # Find your watch's MAC here

    address = "01:E7:65:08:51:E7"  # Replace with your watch's MAC
    async with BleakClient(address) as client:
        print("Connected")

        def handle_notify(_, data):
            bpm = parse_heart_rate(data)
            print(f"Heart Rate: {bpm} bpm")

        await client.start_notify(HR_CHAR_UUID, handle_notify)
        await asyncio.sleep(30)  # Keep reading for 30 seconds
        await client.stop_notify(HR_CHAR_UUID)

asyncio.run(run())

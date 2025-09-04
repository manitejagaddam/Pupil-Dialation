import asyncio
from bleak import BleakScanner, BleakClient

async def explore_all_devices():
    print("🔍 Scanning for devices...\n")
    devices = await BleakScanner.discover()

    if not devices:
        print("❌ No devices found.")
        return

    print(f"✅ Found {len(devices)} devices.\n")

    # Loop through all devices
    for i, d in enumerate(devices):
        print(f"\n[{i}] Trying {d.name or 'Unknown'} ({d.address})...")

        try:
            async with BleakClient(d.address) as client:
                if not client.is_connected:
                    print(f"❌ Could not connect to {d.address}")
                    continue

                print(f"✅ Connected to {d.name or 'Unknown'} ({d.address})")
                services = await client.get_services()

                for service in services:
                    print(f"\n📡 Service: {service.uuid}")
                    for char in service.characteristics:
                        print(f"   └── Char: {char.uuid} (props: {char.properties})")

                print("\n🔍 Done with this device.\n")
        except Exception as e:
            print(f"⚠️ Skipped {d.address} ({d.name}): {e}")

asyncio.run(explore_all_devices())

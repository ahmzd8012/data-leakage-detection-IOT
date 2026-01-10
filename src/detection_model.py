#!/usr/bin/env python3
"""
Smart Home Security - Detection Model
Simple version for testing
"""

print("=" * 50)
print("Smart Home Security Detection System")
print("=" * 50)

# Create sample data
data = [
    {"time": 1.0, "size": 512, "type": "normal"},
    {"time": 1.5, "size": 1024, "type": "normal"},
    {"time": 2.0, "size": 600, "type": "attack"},
    {"time": 2.5, "size": 4000, "type": "attack"},
    {"time": 3.0, "size": 800, "type": "normal"},
]

print("\n📊 Packet Analysis:")
print("-" * 50)

for i, packet in enumerate(data, 1):
    if packet["size"] > 3000:
        detection = "🔴 MALICIOUS (Large packet)"
    elif packet["size"] < 700:
        detection = "🟡 SUSPICIOUS (Small packet)"
    else:
        detection = "🟢 NORMAL"
    
    actual = "ATTACK" if packet["type"] == "attack" else "NORMAL"
    
    print(f"Packet {i}:")
    print(f"  Time: {packet['time']}s | Size: {packet['size']} bytes")
    print(f"  Actual: {actual}")
    print(f"  Detected: {detection}")
    print()

print("=" * 50)
print("✅ Analysis Complete!")
print(f"📈 Total packets analyzed: {len(data)}")
print("=" * 50)
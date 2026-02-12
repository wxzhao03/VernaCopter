import sys

print("Testing motioncapture installation...")

try:
    import motioncapture
    print(f"✓ Version: {motioncapture.__version__}")
    print(f"✓ Location: {motioncapture.__file__}")
    
    print("\nTrying to connect to Vicon...")
    mc = motioncapture.connect("vicon", {'hostname': '131.155.34.241'})
    print("✓ Vicon connection successful!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print(f"Error type: {type(e).__name__}")
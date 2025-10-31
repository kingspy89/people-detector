#!/usr/bin/env python3
"""
IntelliSight AI Analytics - Quick Start Launcher
One-click setup and launch
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    print("\n" + "="*70)
    print("🎯 IntelliSight AI Customer Analytics Platform")
    print("="*70)
    print("🚀 Individual Customer Tracking & Business Intelligence")
    print("📊 Beautiful Dashboards & Advanced Analytics")
    print("="*70)

def check_requirements():
    print("\n🔍 Checking requirements...")

    required = ['cv2', 'pandas', 'streamlit', 'plotly', 'ultralytics', 'torch']
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - missing")

    if missing:
        print(f"\n🔧 Installing missing packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

    return True

def setup_directories():
    print("\n📁 Setting up directories...")
    for dir_name in ['data', 'logs', 'output', 'exports', 'screenshots']:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ {dir_name}/")

def main_menu():
    while True:
        print("\n" + "="*60)
        print("🎯 IntelliSight Main Menu")
        print("="*60)
        print("\n1. 🎬 Start Customer Tracking")
        print("2. 📊 Launch Business Dashboard")
        print("3. 🎮 Run System Demo")
        print("4. 📚 View Documentation")
        print("5.  Exit")
        print("\n" + "="*60)

        choice = input("\n👆 Enter choice (1-5): ").strip()

        if choice == '1':
            print("\n🎬 Customer Tracking Options:")
            print("1. Use camera (live)")
            print("2. Use video file")

            sub = input("Choose (1-2): ").strip()

            if sub == '1':
                subprocess.run([sys.executable, "src/customer_tracking_engine.py", "--source", "0"])
            else:
                video = input("Enter video path: ").strip()
                if video:
                    subprocess.run([sys.executable, "src/customer_tracking_engine.py", "--source", video])

        elif choice == '2':
            print("\n📊 Launching Business Dashboard...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "src/business_dashboard.py"])

        elif choice == '3':
            print("\n🎮 Running Demo...")
            print("✅ Demo: AI Detection → Customer Tracking → Analytics")
            print("💡 Use options 1 or 2 for real analytics")

        elif choice == '4':
            print("\n📚 Documentation:")
            print("   • README.md - Quick start guide")
            print("   • docs/SETUP_GUIDE.md - Detailed setup")
            print("   • docs/USER_MANUAL.md - User guide")

        elif choice == '5':
            print("\n👋 Thank you for using IntelliSight!")
            break

        input("\n⏸️  Press Enter to continue...")

def main():
    print_banner()
    setup_directories()

    if check_requirements():
        main_menu()
    else:
        print("\n❌ Setup failed. Please check requirements.")

if __name__ == "__main__":
    main()

"""
Script to setup Hugging Face Space for AI Trading Terminal
"""
import os
from huggingface_hub import HfApi

def create_space():
    api = HfApi()
    
    # Space configuration
    space_config = {
        "title": "AI Trading Terminal - Advanced Gold Analysis",
        "sdk": "streamlit",
        "emoji": "📈",
        "colorFrom": "yellow",
        "colorTo": "gray",
        "app_file": "app.py",
        "pinned": False,
        "hardware": "cpu-basic",
        "private": False
    }
    
    try:
        # Create or update space
        api.create_space(
            repo_id="aliranjbar777/golden",
            space_sdk="streamlit",
            space_hardware="cpu-basic",
            private=False
        )
        print("✅ Space created/updated successfully!")
        
        # Upload files
        api.upload_file(
            path_or_fileobj="app.py",
            path_in_repo="app.py",
            repo_id="aliranjbar777/golden",
            repo_type="space"
        )
        
        api.upload_file(
            path_or_fileobj="requirements.txt",
            path_in_repo="requirements.txt",
            repo_id="aliranjbar777/golden",
            repo_type="space"
        )
        
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id="aliranjbar777/golden",
            repo_type="space"
        )
        
        print("📁 Files uploaded successfully!")
        print("🌐 Your app will be available at: https://aliranjbar777-golden.hf.space")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try manually at: https://huggingface.co/spaces")

if __name__ == "__main__":
    create_space()

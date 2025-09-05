#!/usr/bin/env python3
"""
BrainBox Unified Launcher
========================

ONE interface to rule them all! 
- Chat with your AI brain
- Watch neural processes visualize
- Configure settings
- Search memory
- Everything in one place!
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import subprocess
import sys
import os
from pathlib import Path
import webbrowser

class BrainBoxLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BrainBox - Unified AI Brain System")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        self.setup_ui()
        self.check_api_keys()
        
    def configure_styles(self):
        """Configure dark theme styles"""
        self.style.configure('Dark.TFrame', background='#1a1a1a')
        self.style.configure('Dark.TLabel', background='#1a1a1a', foreground='#ffffff')
        self.style.configure('Dark.TButton', 
                           background='#2d2d2d', 
                           foreground='#ffffff',
                           borderwidth=1,
                           focuscolor='#4a9eff')
        self.style.map('Dark.TButton',
                      background=[('active', '#3d3d3d'), ('pressed', '#1d1d1d')])
        
    def setup_ui(self):
        """Setup the main UI"""
        
        # Header
        header_frame = ttk.Frame(self.root, style='Dark.TFrame')
        header_frame.pack(fill='x', padx=20, pady=10)
        
        title_label = ttk.Label(header_frame, text="BrainBox AI Brain System", 
                               font=('Arial', 16, 'bold'), style='Dark.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, 
                                 text="Your complete emotional intelligence AI with business ethics", 
                                 font=('Arial', 10), style='Dark.TLabel')
        subtitle_label.pack()
        
        # Main content area with tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Chat Tab
        self.setup_chat_tab(notebook)
        
        # Visualization Tab  
        self.setup_viz_tab(notebook)
        
        # Settings Tab
        self.setup_settings_tab(notebook)
        
        # Status bar
        self.setup_status_bar()
        
    def setup_chat_tab(self, notebook):
        """Setup the main chat interface"""
        chat_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(chat_frame, text='Chat')
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            height=20,
            bg='#2d2d2d',
            fg='#ffffff',
            font=('Consolas', 10),
            wrap='word'
        )
        self.chat_display.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Input area
        input_frame = ttk.Frame(chat_frame, style='Dark.TFrame')
        input_frame.pack(fill='x', padx=10, pady=5)
        
        self.chat_input = tk.Entry(
            input_frame,
            bg='#2d2d2d',
            fg='#ffffff',
            font=('Arial', 11),
            insertbackground='#ffffff'
        )
        self.chat_input.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.chat_input.bind('<Return>', self.send_message)
        
        send_btn = ttk.Button(input_frame, text="Send", command=self.send_message, style='Dark.TButton')
        send_btn.pack(side='right')
        
        # Initialize chat
        self.add_chat_message("BrainBox", "Your AI brain is ready! Ask me anything and I'll show you how I route your query through the Madugu emotional intelligence system.", "system")
        self.add_chat_message("BrainBox", "Try: 'I'm feeling overwhelmed' or 'Write marketing copy' or 'What is consciousness?'", "system")
        
    def setup_viz_tab(self, notebook):
        """Setup visualization controls"""
        viz_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(viz_frame, text='Neural Visualizer')
        
        # Description
        desc_label = ttk.Label(viz_frame, 
                              text="Watch your AI think in real-time!\nLike Windows 98 defrag but for neural processes.",
                              style='Dark.TLabel', font=('Arial', 12))
        desc_label.pack(pady=20)
        
        # Launch button
        launch_viz_btn = ttk.Button(viz_frame, 
                                   text="Launch Neural Process Visualizer",
                                   command=self.launch_visualizer,
                                   style='Dark.TButton')
        launch_viz_btn.pack(pady=20)
        
        # Status
        self.viz_status = ttk.Label(viz_frame, text="Click to start visualization", style='Dark.TLabel')
        self.viz_status.pack(pady=10)
        
    def setup_settings_tab(self, notebook):
        """Setup settings and configuration"""
        settings_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(settings_frame, text='Settings')
        
        # API Keys section
        api_frame = ttk.LabelFrame(settings_frame, text="API Keys & Dependencies", style='Dark.TFrame')
        api_frame.pack(fill='x', padx=20, pady=10)
        
        api_info_label = ttk.Label(api_frame, 
                                  text="Configure your own OpenAI, Claude, or Deepseek API keys\n(BYOK - Bring Your Own Keys!)",
                                  style='Dark.TLabel')
        api_info_label.pack(pady=10)
        
        # Button frame
        btn_frame = ttk.Frame(api_frame, style='Dark.TFrame')
        btn_frame.pack(pady=5)
        
        setup_api_btn = ttk.Button(btn_frame, 
                                  text="Setup API Keys",
                                  command=self.setup_api_keys,
                                  style='Dark.TButton')
        setup_api_btn.pack(side='left', padx=(0, 10))
        
        install_deps_btn = ttk.Button(btn_frame,
                                     text="Install AI Libraries", 
                                     command=self.install_dependencies,
                                     style='Dark.TButton')
        install_deps_btn.pack(side='left')
        
        self.api_status = ttk.Label(api_frame, text="", style='Dark.TLabel')
        self.api_status.pack(pady=5)
        
        # System info
        info_frame = ttk.LabelFrame(settings_frame, text="System Info", style='Dark.TFrame')
        info_frame.pack(fill='x', padx=20, pady=10)
        
        self.system_info = ttk.Label(info_frame, text="Loading system info...", style='Dark.TLabel')
        self.system_info.pack(pady=10)
        
        # Load system info
        threading.Thread(target=self.load_system_info, daemon=True).start()
        
    def setup_status_bar(self):
        """Setup status bar at bottom"""
        status_frame = ttk.Frame(self.root, style='Dark.TFrame')
        status_frame.pack(fill='x', side='bottom')
        
        self.status_label = ttk.Label(status_frame, text="Ready", style='Dark.TLabel')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Version info
        version_label = ttk.Label(status_frame, text="BrainBox v1.0", style='Dark.TLabel')
        version_label.pack(side='right', padx=10, pady=5)
        
    def add_chat_message(self, sender, message, msg_type="user"):
        """Add message to chat display"""
        self.chat_display.configure(state='normal')
        
        if msg_type == "system":
            self.chat_display.insert('end', f"{sender}: {message}\n\n", 'system')
        elif msg_type == "user":
            self.chat_display.insert('end', f"You: {message}\n", 'user')
        else:
            self.chat_display.insert('end', f"{sender}: {message}\n\n", 'assistant')
            
        self.chat_display.configure(state='disabled')
        self.chat_display.see('end')
        
    def send_message(self, event=None):
        """Handle sending chat message"""
        message = self.chat_input.get().strip()
        if not message:
            return
            
        self.chat_input.delete(0, 'end')
        self.add_chat_message("You", message, "user")
        
        # Process message in background thread
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
        
    def process_message(self, message):
        """Process message through BrainBox"""
        try:
            self.status_label.config(text="Processing through Madugu system...")
            
            # Import and use the unified brainbox
            sys.path.append(str(Path(__file__).parent))
            from unified_brainbox import UnifiedBrainBox
            
            # Initialize brain
            brain = UnifiedBrainBox(Path("./brainbox_data"))
            
            # Process query
            result = brain.process_query(message, "auto")
            
            # Show memories found (if any)
            memories = result.get('memories_found', [])
            if memories:
                memory_text = f"🧠 MEMORY SEARCH RESULTS ({len(memories)} found):\n\n"
                for i, memory in enumerate(memories, 1):
                    memory_text += f"Memory {i}: {memory.title}\n"
                    memory_text += f"  {memory.body[:200]}{'...' if len(memory.body) > 200 else ''}\n\n"
                self.add_chat_message("System", memory_text, "system")
            
            # Show response
            response = result['response']
            self.add_chat_message("BrainBox", response, "assistant")
            
            # Show transparency data
            routing = result['routing']
            axiom = result['axiom_review']
            
            transparency = f"Transparency:\n"
            transparency += f"   Emotion: {routing['emotion']} (intensity: {routing['intensity']:.2f})\n"
            transparency += f"   Quadrant: {routing['quadrant']} -> {routing['primary_agent']}\n"
            transparency += f"   AXIOM active: {axiom['axiom_active']}\n"
            
            if axiom['risks_flagged']:
                transparency += f"   Risks flagged: {len(axiom['risks_flagged'])}\n"
                
            transparency += f"   Memory: {result['memory_card']}"
            
            self.add_chat_message("System", transparency, "system")
            
            self.status_label.config(text="Ready")
            
        except Exception as e:
            self.add_chat_message("System", f"Error: {e}", "system")
            self.status_label.config(text="Error occurred")
            
    def launch_visualizer(self):
        """Launch the neural process visualizer"""
        try:
            self.viz_status.config(text="Launching visualizer...")
            subprocess.Popen([sys.executable, "neural_process_visualizer.py"])
            self.viz_status.config(text="Visualizer launched! Check your browser.")
        except Exception as e:
            self.viz_status.config(text=f"Error: {e}")
            messagebox.showerror("Error", f"Failed to launch visualizer: {e}")
            
    def setup_api_keys(self):
        """Launch API key setup"""
        try:
            subprocess.Popen([sys.executable, "setup_your_api.py"])
            self.status_label.config(text="API key setup launched")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch API setup: {e}")
    
    def install_dependencies(self):
        """Launch dependency installer"""
        try:
            subprocess.Popen([sys.executable, "install_dependencies.py"])
            self.status_label.config(text="Dependency installer launched")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch installer: {e}")
            
    def check_api_keys(self):
        """Check if API keys are configured"""
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            self.api_status.config(text="API keys configured")
        else:
            self.api_status.config(text="No API keys found - using placeholder responses")
            
    def load_system_info(self):
        """Load system information"""
        try:
            sys.path.append(str(Path(__file__).parent))
            from unified_brainbox import UnifiedBrainBox
            
            brain = UnifiedBrainBox(Path("./brainbox_data"))
            stats = brain.get_system_stats()
            
            info = f"Node ID: {stats['node_id']}\n"
            info += f"Memory cards: {stats['memory']['total_cards']}\n"
            info += f"Breakfast chain: {'Verified' if stats['breakfast_chain']['verified'] else 'Corrupted'}\n"
            info += f"Components: {len(stats['components'])} active"
            
            self.system_info.config(text=info)
            
        except Exception as e:
            self.system_info.config(text=f"Error loading info: {e}")
            
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    print("Starting BrainBox Unified Launcher...")
    
    app = BrainBoxLauncher()
    app.run()

if __name__ == "__main__":
    main()
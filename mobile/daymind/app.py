"""Kivy entrypoint for the DayMind mobile client."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen

from .audio.recorder import AudioRecorder
from .config import CONFIG
from .services.logger import LogBuffer
from .services.network import ApiClient, ApiError
from .services.uploader import UploadWorker
from .store.queue_store import ChunkQueue
from .store.settings_store import SettingsStore


KV = """
ScreenManager:
    RecordScreen:
        name: "record"
    SummaryScreen:
        name: "summary"
    SettingsScreen:
        name: "settings"

<RecordScreen>:
    BoxLayout:
        orientation: 'vertical'
        ToggleButton:
            id: record_btn
            text: 'Stop Recording' if app.is_recording else 'Start Recording'
            size_hint_y: None
            height: '64dp'
            on_press: app.toggle_recording()
        Label:
            text: '● Recording' if app.is_recording else '◼ Idle'
            color: (1, 0.2, 0.2, 1) if app.is_recording else (0.6, 0.6, 0.6, 1)
            size_hint_y: None
            height: '28dp'
        Label:
            text: f"Queued chunks: {app.queue_size}"
            size_hint_y: None
            height: '32dp'
        ScrollView:
            do_scroll_x: False
            Label:
                text: '\n'.join(app.log_lines)
                size_hint_y: None
                height: self.texture_size[1]
        BoxLayout:
            size_hint_y: None
            height: '48dp'
            Button:
                text: 'Clear Queue'
                on_press: app.clear_queue()
            Button:
                text: 'Summary'
                on_press: app.switch_screen('summary')
            Button:
                text: 'Settings'
                on_press: app.switch_screen('settings')

<SummaryScreen>:
    BoxLayout:
        orientation: 'vertical'
        Label:
            id: summary_label
            text: app.summary_text
            text_size: self.width, None
            halign: 'left'
            valign: 'top'
        Button:
            size_hint_y: None
            height: '48dp'
            text: 'Refresh'
            on_press: app.refresh_summary()
        Button:
            size_hint_y: None
            height: '48dp'
            text: 'Back'
            on_press: app.switch_screen('record')

<SettingsScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: '12dp'
        spacing: '8dp'
        Label:
            text: 'Server URL'
            size_hint_y: None
            height: '24dp'
        TextInput:
            id: server_input
            text: app.server_url
            multiline: False
            size_hint_y: None
            height: '48dp'
        Label:
            text: 'API Key'
            size_hint_y: None
            height: '24dp'
        TextInput:
            id: api_input
            text: app.api_key
            multiline: False
            password: True
            size_hint_y: None
            height: '48dp'
        Button:
            text: 'Save'
            size_hint_y: None
            height: '48dp'
            on_press: app.save_settings(server_input.text, api_input.text)
        Button:
            text: 'Test Connection'
            size_hint_y: None
            height: '48dp'
            on_press: app.test_connection()
        Button:
            text: 'Back'
            size_hint_y: None
            height: '48dp'
            on_press: app.switch_screen('record')
"""


class RecordScreen(Screen):
    pass


class SummaryScreen(Screen):
    pass


class SettingsScreen(Screen):
    pass


class DayMindApp(App):
    is_recording = BooleanProperty(False)
    queue_size = NumericProperty(0)
    summary_text = StringProperty("No summary yet")
    server_url = StringProperty("")
    api_key = StringProperty("")
    log_lines = ListProperty([])

    def build(self):
        Builder.load_string(KV)
        self.base_dir = Path(self.user_data_dir or Path.home() / ".daymind")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.settings_store = SettingsStore(self.base_dir / CONFIG.settings_file)
        self.queue = ChunkQueue(self.base_dir / CONFIG.queue_file)
        self.logger = LogBuffer(CONFIG.log_history)
        self.api_client = ApiClient(self.settings_store)
        self.uploader = UploadWorker(self.queue, self.api_client, self.logger)
        chunks_dir = self.base_dir / "chunks"
        self.recorder = AudioRecorder(chunks_dir, self.logger, self._handle_chunk)
        self._sync_state(initial=True)
        return ScreenManager()

    def on_start(self):
        self.uploader.start()
        Clock.schedule_interval(lambda dt: self._sync_state(), 1)

    def on_stop(self):
        self.recorder.stop()
        self.uploader.stop()
        self.api_client.close()

    def toggle_recording(self):
        if self.is_recording:
            self.recorder.stop()
            self.is_recording = False
        else:
            self.recorder.start()
            self.is_recording = True

    def _handle_chunk(self, path: str) -> None:
        chunk_id = self.queue.enqueue(path)
        self.logger.add(f"Chunk queued ({chunk_id[:6]})")
        self.uploader.wake()
        Clock.schedule_once(lambda dt: self._sync_state(), 0)

    def clear_queue(self):
        self.queue.clear()
        self.logger.add("Queue cleared")
        self._sync_state()

    def refresh_summary(self):
        self.logger.add("Refreshing summary...")
        def worker():
            try:
                summary = self.api_client.fetch_summary()
            except ApiError as exc:
                self.logger.add(str(exc))
                summary = f"Error: {exc}"
            Clock.schedule_once(lambda dt: self._set_summary(summary), 0)

        threading.Thread(target=worker, daemon=True).start()

    def _set_summary(self, text: str):
        self.summary_text = text

    def save_settings(self, server_url: str, api_key: str):
        self.settings_store.update(server_url=server_url.strip(), api_key=api_key.strip())
        self.server_url = self.settings_store.get().server_url
        self.api_key = self.settings_store.get().api_key
        self.logger.add("Settings saved")

    def test_connection(self):
        def worker():
            try:
                ok = self.api_client.test_connection()
                message = "Connection OK" if ok else "Connection failed"
            except ApiError as exc:
                message = f"Connection error: {exc}"
            self.logger.add(message)

        threading.Thread(target=worker, daemon=True).start()

    def switch_screen(self, name: str):
        if self.root:
            self.root.current = name

    def _sync_state(self, initial: bool = False):
        self.queue_size = len(self.queue)
        self.log_lines = self.logger.get()
        settings = self.settings_store.get()
        if initial:
            self.server_url = settings.server_url
            self.api_key = settings.api_key


if __name__ == "__main__":
    DayMindApp().run()

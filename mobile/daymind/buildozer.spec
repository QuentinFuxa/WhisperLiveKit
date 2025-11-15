# Buildozer spec for the DayMind Kivy client.

[app]
title = DayMind
package.name = daymind
package.domain = dev.symbioza
source.dir = .
source.include_exts = py,kv,png,jpg,jpeg,svg,ttf,json,md,wav
version = 0.6.0
requirements = python3,kivy==2.2.1,httpx==0.26.0,sounddevice,numpy
orientation = portrait
fullscreen = 0
log_level = 1
android.api = 33
android.minapi = 24
android.archs = arm64-v8a,armeabi-v7a
android.permissions = RECORD_AUDIO,INTERNET,ACCESS_NETWORK_STATE,WAKE_LOCK,FOREGROUND_SERVICE
android.allow_backup = 0
android.debug = 1
android.extra_args = --no-shrink
android.logcat_filters = *:S python:D

[buildozer]
log_level = 2
warn_on_root = 0


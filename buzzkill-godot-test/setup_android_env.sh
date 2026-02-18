#!/usr/bin/env bash
# Apply Android and Java SDK paths for this project.
# Source before building or opening Godot:  source setup_android_env.sh

# Android SDK (default macOS location; override if yours is elsewhere)
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
export ANDROID_SDK_ROOT="$ANDROID_HOME"

# Java SDK — Godot 4 recommends OpenJDK 17; falls back to JDK 22 or 25 if present
if [[ -n "$JAVA_HOME" ]]; then
  : # already set
elif command -v /usr/libexec/java_home &>/dev/null; then
  for ver in 17 22 25 21 11; do
    JAVA_HOME=$(/usr/libexec/java_home -v "$ver" 2>/dev/null) && break
  done
  export JAVA_HOME
fi

if [[ -z "$JAVA_HOME" ]]; then
  echo "Warning: JAVA_HOME not set. Install OpenJDK 17 for Godot Android export: brew install openjdk@17"
fi
if [[ ! -d "$ANDROID_HOME" ]]; then
  echo "Warning: Android SDK not found at $ANDROID_HOME. Install via Android Studio or command-line tools."
fi

echo "ANDROID_HOME=$ANDROID_HOME"
echo "JAVA_HOME=${JAVA_HOME:-<not set>}"

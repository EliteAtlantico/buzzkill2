# Android & Java SDK setup for Buzzkill Godot (XR)

## 1. Apply SDK paths in your shell (optional)

From the project root (`buzzkill-godot-test/`):

```bash
source setup_android_env.sh
```

This sets `ANDROID_HOME`, `ANDROID_SDK_ROOT`, and `JAVA_HOME` for the current terminal. You can run it before opening Godot or running export scripts.

## 2. Set paths in Godot Editor (required for Android export)

1. Open the project in Godot.
2. **Godot** (macOS) or **Editor** (Windows/Linux) → **Editor Settings**.
3. Open the **Android** section.
4. Set:
   - **Java SDK Path**: your JDK install (e.g. `/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home` or OpenJDK 17 from Homebrew).
   - **Android Sdk Path**: your Android SDK (e.g. `/Users/YourUser/Library/Android/sdk` on macOS).

Godot 4 recommends **OpenJDK 17** for Android export. Install with:

```bash
brew install openjdk@17
```

Then set **Java SDK Path** in Editor Settings to:  
`$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home`

## 3. Install missing Android SDK components (if needed)

Your SDK already has build-tools and platforms. For Godot 4 you also need the **NDK** (e.g. r23c). When you have disk space, run:

```bash
source setup_android_env.sh
$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager --sdk_root="$ANDROID_HOME" \
  "ndk;23.2.8568313" "platforms;android-34" "build-tools;34.0.0"
```

(Installation failed earlier due to “No space left on device” — free space and re-run the command above.)

## 4. Add an Android export preset in Godot

1. **Project → Export…**
2. Click **Add…** → **Android**
3. Configure the preset (package name, icons, etc.) and export.

Once the two paths in Editor Settings are set and the NDK is installed, you can export to APK or AAB from **Project → Export…**.

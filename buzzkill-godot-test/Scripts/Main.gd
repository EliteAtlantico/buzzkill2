extends Node3D

var xr_interface: XRInterface



## Run in VR (headset) or Desktop (window with mouse/keyboard).
## Toggle this in the Inspector on the Main node, or set default below.
@export_enum("VR", "Desktop") var run_mode: int = 1  # 0 = VR, 1 = Desktop

var _xr_initialized: bool = false

func _ready() -> void:
	call_deferred("_setup_xr")
	xr_interface = XRServer.find_interface("OpenXR")
	
	if xr_interface and xr_interface.is_initialized():
		print("OpenXR initialized successfully") 
		
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
		
		get_viewport().use_xr = true
	else:
		print("OpenXR not initialized, check if headset connected")
	
func _setup_xr() -> void:
	var want_vr: bool = (run_mode == 0)

	if not want_vr:
		_use_desktop_mode()
		return

	var xr := XRServer.find_interface("OpenXR")
	if xr == null:
		push_warning("OpenXR interface not found. Using Desktop mode.")
		_use_desktop_mode()
		return

	if not xr.is_initialized():
		if not xr.initialize():
			push_warning("OpenXR failed to initialize. Using Desktop mode.")
			_use_desktop_mode()
			return

	_xr_initialized = true
	get_viewport().use_xr = true
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	var cam := get_node_or_null("Camera3D")
	if cam:
		cam.set_current(false)
		cam.visible = false
	print("Running in VR mode.")

func _use_desktop_mode() -> void:
	get_viewport().use_xr = false
	var cam := get_node_or_null("Camera3D")
	if cam:
		cam.set_current(true)
		cam.visible = true
	var origin := get_node_or_null("XROrigin3D")
	if origin:
		origin.visible = false
	print("Running in Desktop mode.")

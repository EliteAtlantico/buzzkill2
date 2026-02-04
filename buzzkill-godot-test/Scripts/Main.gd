extends Node3D

@export var enable_xr: bool = false

func _enter_tree() -> void:
	if not enable_xr:
		return

	var xr := XRServer.find_interface("OpenXR")
	if xr == null:
		push_error("OpenXR interface missing")
		return

	if not xr.is_initialized():
		if not xr.initialize():
			push_error("OpenXR failed to initialize")
			return

func _ready() -> void:
	if enable_xr:
		get_viewport().use_xr = true
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)

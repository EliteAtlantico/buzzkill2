extends Node3D

@export var enable_xr: bool = false

@onready var _udp_receiver: Node = $UDPReceiver
@onready var _player_logic: Node = $XROrigin3D.get_node("Player Logic")

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
	# Wire Muse / focused-high data into player logic.
	if is_instance_valid(_udp_receiver) and is_instance_valid(_player_logic):
		if not _udp_receiver.is_connected("focus_update", Callable(_player_logic, "_on_focus_update")):
			_udp_receiver.connect("focus_update", Callable(_player_logic, "_on_focus_update"))

	# XR setup stays as before.
	if enable_xr:
		get_viewport().use_xr = true
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)

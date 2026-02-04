extends XRController3D

@export var speed: float = 2.0

var xr: XRInterface

func _ready() -> void:
	xr = XRServer.find_interface("OpenXR")

func _physics_process(delta: float) -> void:
	# Guard 1: OpenXR interface must exist and be initialized
	if xr == null or not xr.is_initialized():
		return

	# Guard 2: This controller must actually be tracked/registered
	# tracker is inherited from XRNode3D; standard names include "left_hand" and "right_hand".
	if XRServer.get_tracker(tracker) == null:
		return

	# Action values are defined by the active OpenXR action map.
	var move: Vector2 = get_vector2(&"move")
	if move.length() < 0.1:
		return

	var forward: Vector3 = -global_transform.basis.z
	get_parent().translate(forward * move.y * speed * delta)

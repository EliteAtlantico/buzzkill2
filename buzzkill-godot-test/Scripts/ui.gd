extends Control

@onready var bar: ProgressBar = $ProgressBar

var _player_logic: Node = null

func _ready() -> void:
	# The UI scene instances the main 3D scene as a child called "Main".
	# From there we drill down to the "Player Logic" node that holds the damage multiplier.
	if has_node("Main/XROrigin3D/Player Logic"):
		_player_logic = get_node("Main/XROrigin3D/Player Logic")
	else:
		push_error("UI.gd: Could not find 'Main/XROrigin3D/Player Logic' – damage meter will not update.")

func _process(_delta: float) -> void:
	if _player_logic == null:
		return

	# Read the damage multiplier driven by Muse / focused-high data
	# and map it into 0..1 for the progress bar visualization.
	var min_mult: float = _player_logic.min_damage_multiplier
	var max_mult: float = _player_logic.max_damage_multiplier
	var dmg_mult: float = _player_logic.damage_multiplier

	var span := max_mult - min_mult
	var normalized := 0.0
	if span > 0.001:
		normalized = (dmg_mult - min_mult) / span

	bar.value = clamp(normalized, 0.0, 1.0)

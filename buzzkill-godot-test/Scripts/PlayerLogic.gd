extends Node

# Live focused-high inputs coming from the Muse / LSL pipeline.
var focused_high_score: float = 0.0
var focused_high_meter: float = 0.0

# How quickly the focused-high meter climbs and falls.
@export var rise: float = 1.0
@export var decay: float = 0.5

# Damage multiplier that other gameplay systems can read.
# This will be driven by how "high" the focused-high meter is.
@export var min_damage_multiplier: float = 0.5
@export var max_damage_multiplier: float = 3.0

var damage_multiplier: float = 1.0

func _process(delta: float) -> void:
	# Smooth the focused-high meter locally (in case upstream is noisy).
	focused_high_meter = clamp(
		focused_high_meter
		+ focused_high_score * rise * delta
		- (1.0 - focused_high_score) * decay * delta,
		0.0,
		1.0
	)

	# Map the 0..1 focused-high meter into a tunable damage multiplier range.
	var span := max_damage_multiplier - min_damage_multiplier
	if span <= 0.001:
		damage_multiplier = min_damage_multiplier
	else:
		damage_multiplier = min_damage_multiplier + focused_high_meter * span

func _on_focus_update(score: float, meter: float) -> void:
	focused_high_score = score
	focused_high_meter = meter

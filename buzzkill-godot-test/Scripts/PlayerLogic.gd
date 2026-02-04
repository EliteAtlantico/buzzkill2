extends Node

var focused_high_score: float = 0.0
var focused_high_meter: float = 0.0

@export var rise: float = 1.0
@export var decay: float = 0.5

func _process(delta: float) -> void:
	focused_high_meter = clamp(
		focused_high_meter
		+ focused_high_score * rise * delta
		- (1.0 - focused_high_score) * decay * delta,
		0.0,
		1.0
	)

func _on_focus_update(score: float, meter: float) -> void:
	focused_high_score = score
	focused_high_meter = meter

extends Node

signal focus_update(score: float, meter: float)

var udp: PacketPeerUDP

func _ready() -> void:
	udp = PacketPeerUDP.new()
	udp.bind(12000)

func _process(_delta: float) -> void:
	while udp.get_available_packet_count() > 0:
		var packet: String = udp.get_packet().get_string_from_utf8()

		var parsed: Variant = JSON.parse_string(packet)
		if typeof(parsed) != TYPE_DICTIONARY:
			continue

		var data: Dictionary = parsed
		if not data.has("focused_high"):
			continue

		var fh: Dictionary = data["focused_high"]

		var score: float = float(fh.get("score", 0.0))
		var meter: float = float(fh.get("meter", 0.0))

		emit_signal("focus_update", score, meter)

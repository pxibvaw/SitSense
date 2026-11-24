package com.example.barunjari

import org.json.JSONObject

internal const val PostureAlertMinutes = 1

internal enum class AlertPosture(val subject: String, val advice: String) {
    NECK("거북목 자세가", "목과 어깨를 가볍게 움직여 보세요"),
    CROSSED_LEGS("다리 꼬기가", "다리를 풀고 양발을 바닥에 두세요"),
    LEAN("한쪽으로 기대는 자세가", "체중을 좌우에 고르게 실어 보세요"),
    SITTING("연속 착석이", "자리에서 일어나 가볍게 걸어 보세요")
}

internal data class PostureAlert(val posture: AlertPosture, val minutes: Int) {
    val title: String get() = "${posture.subject} ${minutes}분째 이어지고 있어요"
}

internal fun JSONObject.alertPostures(): Map<AlertPosture, String> = buildMap {
    val seat = optString("seat_state")
    if (!optBoolean("seated") || seat == "empty") return@buildMap
    put(AlertPosture.SITTING, "seated")
    val head = optString("head_state")
    if (liveUpperBodyLabel(head, optBoolean("pose_detected"), optJSONObject("confidence")
            ?.optDouble("head")?.takeIf { it.isFinite() }) == "거북목 자세") {
        put(AlertPosture.NECK, "forward_head")
    }
    when (seat) {
        "cross_left", "cross_right" -> put(AlertPosture.CROSSED_LEGS, seat)
        "lean_left", "lean_right" -> put(AlertPosture.LEAN, seat)
    }
}

internal class PostureAlertTracker {
    private data class Episode(val signal: String, val start: Double, var dismissed: Boolean = false)

    private val episodes = mutableMapOf<AlertPosture, Episode>()
    private var lastSample: Double? = null

    fun record(timestamp: Double, active: Map<AlertPosture, String>) {
        if (!timestamp.isFinite()) {
            reset()
            return
        }
        // A clock adjustment or a gap in samples breaks continuity.
        if (lastSample?.let { timestamp < it || timestamp - it > 10 } == true) reset()
        lastSample = timestamp
        episodes.keys.retainAll(active.keys)
        active.forEach { (posture, signal) ->
            if (episodes[posture]?.signal != signal) episodes[posture] = Episode(signal, timestamp)
        }
    }

    fun alert(): PostureAlert? {
        val now = lastSample ?: return null
        return AlertPosture.entries.firstNotNullOfOrNull { posture ->
            val episode = episodes[posture] ?: return@firstNotNullOfOrNull null
            val elapsed = now - episode.start
            if (episode.dismissed || elapsed < PostureAlertMinutes * 60) null
            else PostureAlert(posture, (elapsed / 60).toInt())
        }
    }

    fun dismiss(posture: AlertPosture) {
        episodes[posture]?.dismissed = true
    }

    fun reset() {
        episodes.clear()
        lastSample = null
    }
}

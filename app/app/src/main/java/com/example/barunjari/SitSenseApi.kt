package com.example.barunjari

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URI

internal const val DefaultServerAddress = "http://172.20.99.84:28473"
internal const val LivePostureWaiting = "판정 대기 중"

internal class SitSenseResponseException(val statusCode: Int, val errorCode: String?) :
    IllegalStateException("서버 응답 $statusCode${errorCode?.let { ": $it" }.orEmpty()}")

internal class SitSenseApi(address: String) {
    private val base = address.trim().trimEnd('/').also {
        val uri = URI(it)
        require(uri.scheme in listOf("http", "https") && !uri.host.isNullOrBlank() &&
            uri.userInfo == null && uri.query == null && uri.fragment == null && uri.path.isNullOrEmpty()) {
            "서버 주소를 $DefaultServerAddress 형식으로 입력해 주세요"
        }
    }

    suspend fun request(path: String, body: JSONObject? = null, readTimeoutMillis: Int = 4000): JSONObject = withContext(Dispatchers.IO) {
        val connection = URI(base + path).toURL().openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 4000
            connection.readTimeout = readTimeoutMillis
            connection.instanceFollowRedirects = false
            connection.setRequestProperty("Accept", "application/json")
            if (body != null) {
                connection.requestMethod = "POST"
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                connection.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
            }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val result = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                val errorCode = runCatching { JSONObject(result).optJSONObject("detail")?.optString("code") }
                    .getOrNull()?.takeIf { it.isNotBlank() }
                throw SitSenseResponseException(code, errorCode)
            }
            JSONObject(result)
        } finally { connection.disconnect() }
    }

    suspend fun sync() = request("/time/sync", JSONObject().put("app_time", System.currentTimeMillis() / 1000.0).put("timezone", "Asia/Seoul"))

    suspend fun calibrate(position: SeatCalibrationPosition? = null) = request(
        position?.let { "/calibrate?position=${it.apiValue}" } ?: "/calibrate", JSONObject()
    )

    suspend fun resetAllRecords() {
        try {
            val result = request("/records/reset", JSONObject().put("confirm", "DELETE_ALL_RECORDS"), readTimeoutMillis = 30_000)
            check(result.optJSONObject("deleted")?.has("baselines") == true) {
                "기준 자세까지 삭제됐는지 확인할 수 없어요. Pi 서버 버전을 확인해 주세요."
            }
        } catch (e: SitSenseResponseException) {
            if (e.statusCode == 404 || e.errorCode == "CONFIRM_REQUIRED") {
                error("전체 초기화를 사용하려면 Pi 서버를 업데이트해 주세요.")
            }
            throw e
        }
    }
}

internal enum class SeatCalibrationPosition(val apiValue: String, val title: String, val instruction: String) {
    FRONT("front", "앞", "의자 앞쪽에 바르게 앉아 주세요."),
    BACK("back", "뒤", "등받이까지 깊게 앉아 주세요.");

    val kind: String get() = "position_$apiValue"
}

internal fun JSONObject.calibratedSeatPositions(): Set<SeatCalibrationPosition> {
    val positions = optJSONArray("positions_done") ?: return emptySet()
    val names = (0 until positions.length()).map { positions.optString(it) }.toSet()
    return SeatCalibrationPosition.entries.filterTo(mutableSetOf()) { it.apiValue in names }
}

internal fun JSONObject.calibrationSnapshot(): CalibrationSnapshot = CalibrationSnapshot(
    state = optString("state", "none"),
    kind = optString("kind"),
    hasMiddle = !isNull("measured_at"),
    supportsPositions = has("positions_done"),
    positionsDone = calibratedSeatPositions()
)

internal fun baselineMessage(state: String, reason: String): String = when (state) {
    "waiting_seat" -> "방석에 앉아 주세요"
    "measuring" -> "바른 자세를 유지해 주세요"
    "ready" -> "기준 자세 측정이 완료됐어요"
    "failed" -> when (reason) {
        "pressure_low" -> "방석에 깊숙이 바르게 앉아 다시 측정해 주세요"
        "no_person" -> "카메라에 상체가 보이도록 앉아 주세요"
        "sensor_lost" -> "센서 연결을 확인하고 다시 측정해 주세요"
        "too_much_motion" -> "움직임을 줄이고 다시 측정해 주세요"
        else -> "측정에 실패했어요. 다시 시도해 주세요"
    }
    else -> "기준 자세를 측정해 주세요"
}

internal fun seatSideLabel(value: String): String? = when (value) {
    "left" -> "왼쪽 (확정)"
    "right" -> "오른쪽 (확정)"
    else -> null
}

internal fun seatKindLabel(value: String, fallback: String = "판정 불가"): String = when (value) {
    "cross_left", "cross_right" -> "다리 꼬기 (추정)"
    "lean_left", "lean_right" -> "체중 편향 (추정)"
    else -> postureName(value, fallback)
}

internal fun liveUpperBodyLabel(value: String, poseDetected: Boolean, confidence: Double?): String =
    if (poseDetected && confidence != 0.0) postureName(value, LivePostureWaiting) else LivePostureWaiting

internal fun postureName(value: String, fallback: String = "판정 불가"): String = when (value) {
    "normal" -> "정상 자세"; "none" -> "기울기 없음"; "forward", "forward_head" -> "거북목 자세"
    "cross_left" -> "왼다리 꼬기 (추정)"; "cross_right" -> "오른다리 꼬기 (추정)"
    "lean_left" -> "왼쪽 체중 편향 (추정)"; "lean_right" -> "오른쪽 체중 편향 (추정)"
    "tilt_left", "left" -> "왼쪽 기울기"; "tilt_right", "right" -> "오른쪽 기울기"
    "empty" -> "자리 비움"; "ok" -> "적정 거리"; "caution" -> "거리 주의"; "warning" -> "너무 가까워요"
    else -> fallback
}

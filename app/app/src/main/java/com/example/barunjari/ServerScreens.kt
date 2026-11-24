package com.example.barunjari

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.Image
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.clip
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CancellationException
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.roundToInt

private val ServerGreen = Color(0xFF16A34A)
private val ServerMuted = Color(0xFF6B7280)
private val ChartTrack = Color(0xFFE5E7EB)
private val ChartMint = Color(0xFFE7F6EC)
private val GoalKeys = listOf("cross", "forward_head", "lean", "long_sitting")
private fun JSONObject.number(key: String): Double? = if (isNull(key)) null else optDouble(key).takeIf { it.isFinite() }
private fun JSONArray.number(index: Int): Double? = optDouble(index).takeIf { it.isFinite() }
private fun JSONArray?.records(): List<JSONObject> = if (this == null) emptyList() else (0 until length()).mapNotNull(::optJSONObject)
private fun seconds(value: Double?): String = value?.let { "${(it / 60).toInt()}분 ${it.toInt() % 60}초" } ?: "기록 없음"
private fun percent(value: Double?): String = value?.let { "${(it * 100).roundToInt()}%" } ?: "판정 기록 없음"
internal fun distanceLabel(millimeters: Double?): String? = millimeters?.takeIf { it.isFinite() }?.let {
    String.format(Locale.KOREA, "%.1fcm", it / 10)
}

@Composable internal fun ServerDevice(status: JSONObject, connected: Boolean, busy: Boolean, toggle: () -> Unit) {
    ServerColumn {
        Label("기기 연결", 24, weight = FontWeight.Bold)
        Label(if (connected) "${status.optString("mode")} 서버 연결" else "기기를 연결해 주세요", 18)
        val sensors = status.optJSONObject("sensors")
        listOf("camera" to "카메라", "pressure" to "방석 압력 센서", "distance" to "적외선 거리 센서").forEach { (key, name) ->
            val sensor = sensors?.optJSONObject(key)
            DesignCard(Modifier.fillMaxWidth()) {
                Label(name, 17, weight = FontWeight.Bold)
                Label(if (!connected) "연결 대기" else if (sensor?.optBoolean("ok") == true) "정상 응답" else "연결 확인 필요", 14)
            }
        }
        PrimaryButton(if (connected) "연결 끊기" else "서버 주소로 연결", toggle, enabled = !busy, color = ServerGreen)
        Label("주변 기기 자동 검색은 서버 API가 아직 제공되지 않았어요", 12, ServerMuted)
    }
}

@Composable private fun ServerColumn(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp), content = content)
}

@Composable
private fun ServerSetupColumn(
    step: Int?,
    onBack: () -> Unit,
    backEnabled: Boolean = true,
    content: @Composable ColumnScope.() -> Unit
) {
    BackHandler { if (backEnabled) onBack() }
    Column(
        Modifier.fillMaxSize().background(Color.White).verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        StepHeader(step, onBack, total = 3, backEnabled = backEnabled)
        content()
    }
}

@Composable
internal fun ServerConnect(
    address: String,
    change: (String) -> Unit,
    busy: Boolean,
    error: String,
    showSetupStep: Boolean,
    onBack: () -> Unit,
    onConnect: () -> Unit
) {
    val focusManager = LocalFocusManager.current
    ServerSetupColumn(if (showSetupStep) 2 else null, onBack, backEnabled = !busy) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Label("기기와 센서를 연결해 주세요", 25, weight = FontWeight.Bold)
            Label("Pi와 같은 Wi-Fi에 연결한 뒤 서버 주소를 입력해 주세요.", 14, ServerMuted)
        }
        Image(
            painterResource(R.drawable.design_pairing),
            contentDescription = null,
            modifier = Modifier.align(Alignment.CenterHorizontally).widthIn(max = 372.dp)
                .fillMaxWidth().aspectRatio(372f / 184f)
        )
        OutlinedTextField(
            value = address,
            onValueChange = change,
            label = { Text("서버 주소") },
            singleLine = true,
            enabled = !busy,
            shape = RoundedCornerShape(16.dp),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
            modifier = Modifier.fillMaxWidth()
        )
        PrimaryButton(
            if (busy) "연결 확인 중…" else "기기 연결",
            onClick = { focusManager.clearFocus(); onConnect() },
            enabled = !busy,
            color = ServerGreen
        )
        if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
        if (error.isNotBlank()) {
            Label(error, 13, MaterialTheme.colorScheme.error,
                modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
        }
    }
}

@Composable
internal fun ServerCalibration(
    session: JSONObject,
    busy: Boolean,
    showSetupStep: Boolean,
    onBack: () -> Unit,
    onMeasure: suspend (CalibrationStep) -> CalibrationSnapshot,
    onDone: () -> Unit
) {
    val baseline = session.optJSONObject("baseline") ?: JSONObject()
    val snapshot = baseline.calibrationSnapshot()
    val latestMeasure by rememberUpdatedState(onMeasure)
    val latestDone by rememberUpdatedState(onDone)
    var step by rememberSaveable { mutableStateOf(CalibrationStep.MIDDLE) }
    var attempt by rememberSaveable { mutableIntStateOf(0) }
    var failed by rememberSaveable { mutableStateOf(false) }
    var canSkip by rememberSaveable { mutableStateOf(false) }
    var error by rememberSaveable { mutableStateOf("") }

    LaunchedEffect(attempt) {
        if (failed) return@LaunchedEffect
        try {
            while (true) {
                val result = latestMeasure(step)
                if (!result.completed(step)) {
                    failed = true
                    canSkip = result.state == "failed" && result.hasMiddle && step.position != null
                    if (result.state != "failed") error = "기준 측정 완료를 확인하지 못했어요. 다시 시도해 주세요."
                    return@LaunchedEffect
                }
                val next = result.nextPosition()
                if (next == null) {
                    latestDone()
                    return@LaunchedEffect
                }
                step = next
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            failed = true
            error = e.message ?: "서버에 연결할 수 없어요"
        }
    }

    val currentStep = step.matches(snapshot.kind)
    ServerSetupColumn(if (showSetupStep) 3 else null, onBack) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Label("기준 자세 설정", 25, weight = FontWeight.Bold)
            Label(step.position?.instruction ?: "바른 자세로 편하게 앉아 주세요.", 14, ServerMuted)
        }
        if (currentStep) {
            CalibrationProgress(baseline, busy)
            if (snapshot.inProgress || (failed && snapshot.state == "failed")) {
                CalibrationMessage(baseline, pressureLowMessage = step.position?.let {
                    "양쪽에 체중을 고르게 싣고 다시 측정해 주세요."
                })
            }
        } else if (busy) {
            LinearProgressIndicator(Modifier.fillMaxWidth())
        }
        if (error.isNotBlank()) {
            Label(error, 13, MaterialTheme.colorScheme.error,
                modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
        }
        if (failed) {
            PrimaryButton("다시 측정", {
                failed = false
                canSkip = false
                error = ""
                attempt++
            }, enabled = !busy, color = ServerGreen)
            if (canSkip) TextButton(onClick = onDone, enabled = !busy) { Text("건너뛰기 (가운데 기준 사용)") }
        }
        PostureSetupGuide()
    }
}

@Composable
internal fun ServerPositionCalibration(
    session: JSONObject,
    busy: Boolean,
    error: String,
    onBack: () -> Unit,
    onStart: (SeatCalibrationPosition) -> Unit
) {
    val baseline = session.optJSONObject("baseline") ?: JSONObject()
    val state = baseline.optString("state")
    val paused = session.optString("state") == "paused"
    val inProgress = !paused && (state == "waiting_seat" || state == "measuring")
    val hasMiddle = baseline.number("measured_at") != null
    val completed = baseline.calibratedSeatPositions()
    val next = SeatCalibrationPosition.entries.firstOrNull { it !in completed }
    val measuringPosition = SeatCalibrationPosition.entries.firstOrNull { "position_${it.apiValue}" == baseline.optString("kind") }
    val failed = next != null && next == measuringPosition && state == "failed"

    ServerSetupColumn(null, onBack, backEnabled = !busy) {
        Label("앞·뒤 기준 추가 측정", 25, weight = FontWeight.Bold)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SeatCalibrationPosition.entries.forEach { step ->
                Pill("${step.title} ${if (step in completed) "완료" else "측정 전"}", green = step in completed)
            }
        }
        when {
            paused -> Label("측정을 재개한 뒤 앞·뒤 기준을 추가할 수 있어요.", 14, ServerMuted)
            inProgress -> measuringPosition?.let { Label(it.instruction, 14, ServerMuted) }
            !hasMiddle -> Label("설정에서 기준 자세를 먼저 측정해 주세요.", 14, ServerMuted)
            next != null -> Label("${next.instruction} 위치마다 10초씩 측정해요.", 14, ServerMuted)
            else -> Label("앞·뒤 기준 측정이 완료됐어요.", 16, ServerGreen, FontWeight.Bold)
        }
        if (!paused) {
            CalibrationProgress(baseline, busy)
            if (inProgress || failed) {
                CalibrationMessage(baseline, pressureLowMessage = "양쪽에 체중을 고르게 싣고 다시 측정해 주세요.")
            }
        }
        if (error.isNotBlank()) {
            Label(error, 13, MaterialTheme.colorScheme.error,
                modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
        }
        if (!inProgress) {
            if (next == null || !hasMiddle || paused) {
                PrimaryButton("설정으로", onBack, enabled = !busy, color = ServerGreen)
            } else {
                PrimaryButton(if (failed) "${next.title} 기준 다시 측정" else "${next.title} 기준 측정 시작",
                    { onStart(next) }, enabled = !busy, color = ServerGreen)
            }
        }
    }
}

@Composable
private fun ColumnScope.CalibrationProgress(baseline: JSONObject, busy: Boolean) {
    val state = baseline.optString("state")
    val secondsLeft = baseline.number("remaining_sec")?.let { ceil(it).toInt() }
    if (state == "measuring" && secondsLeft != null) {
        Box(Modifier.align(Alignment.CenterHorizontally).size(160.dp), contentAlignment = Alignment.Center) {
            CircularProgressIndicator(Modifier.fillMaxSize(), color = ServerGreen,
                strokeWidth = 10.dp, trackColor = ServerGreen.copy(alpha = .12f))
            Label("${secondsLeft}초 남음", 24, ServerGreen, FontWeight.Bold)
        }
    } else if (state == "measuring" || state == "waiting_seat" || busy) {
        LinearProgressIndicator(Modifier.fillMaxWidth())
    }
}

@Composable
private fun CalibrationMessage(baseline: JSONObject, pressureLowMessage: String? = null) {
    val state = baseline.optString("state")
    val reason = baseline.optString("fail_reason")
    val message = if (state == "failed" && reason == "pressure_low") pressureLowMessage else null
    Label(message ?: baselineMessage(state, reason), 16,
        color = if (state == "failed") MaterialTheme.colorScheme.error else ServerGreen,
        weight = FontWeight.Bold, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
}

@Composable
internal fun ServerHome(
    status: JSONObject,
    current: JSONObject,
    layout: JSONObject,
    report: JSONObject,
    error: String,
    postureAlert: PostureAlert? = null,
    dismissPostureAlert: (AlertPosture) -> Unit = {},
    openFeedback: () -> Unit
) {
    ServerColumn {
        Column {
            Label(ReportDates.format(ReportDates.today(), "M월 d일 EEEE"), 13, ServerMuted)
            Label("바른자리", 24, weight = FontWeight.Bold)
        }
        if (status.optString("mode") == "mock") Label("예시 데이터 · mock 서버", 13, ServerGreen)
        if (error.isNotBlank()) Label("연결 확인 필요 · $error", 13, Color(0xFFE85050))
        DesignCard(Modifier.fillMaxWidth()) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Pill("하루 평균 자세")
                    val posture = report.optJSONObject("daily_posture")?.optString("posture")
                    Label(posture?.let { seatKindLabel(it) } ?: "기록 없음", 24, weight = FontWeight.Bold)
                }
                Image(painterResource(R.drawable.design_summary_plant), null, Modifier.size(86.dp))
            }
        }
        Row(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(Color(0xFF12813A)).padding(18.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            val ratio = report.number("normal_ratio")
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Label("오늘 정상 자세 비율", 14, Color.White)
                Label(percent(ratio), if (ratio == null) 18 else 40, Color.White, FontWeight.Bold)
                Label("총 착석 ${seconds(report.number("seated_sec"))} · 정상 ${seconds(report.number("normal_sec"))}", 12, Color.White)
                Label("판정 불가 시간은 비율 계산에서 제외해요", 11, Color.White)
            }
            if (ratio != null) ProgressRing((ratio * 100).toFloat().coerceIn(0f, 100f), Modifier.size(74.dp),
                color = Color.White, base = Color.White.copy(alpha = .28f))
        }
        postureAlert?.let { alert -> PostureAlertBanner(alert) { dismissPostureAlert(alert.posture) } }
        Row(Modifier.fillMaxWidth().padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Label("지금 자세", 17, weight = FontWeight.Bold, modifier = Modifier.weight(1f))
            Label("상체·하체를 따로 기록해요", 12, FeedbackMuted)
        }
        LivePostureCards(current, layout)
        val goals = report.optJSONArray("goals").records()
        if (goals.isNotEmpty()) DesignCard(Modifier.fillMaxWidth()) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Label("내 목표", 17, weight = FontWeight.Bold, modifier = Modifier.weight(1f))
                TextButton(onClick = openFeedback) { Text("피드백 보기") }
            }
            goals.forEachIndexed { index, goal ->
                if (index > 0) HorizontalDivider(color = ChartTrack)
                Row(Modifier.fillMaxWidth().padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    GoalIcon(goal.optString("goal"))
                    Column(Modifier.weight(1f)) {
                        Label(goalResultLabel(goal), 14, weight = FontWeight.SemiBold)
                        Label("오늘 ${seconds(goal.number("today"))}", 12, ServerMuted)
                    }
                }
            }
        }
    }
}

@Composable
internal fun PostureAlertBanner(alert: PostureAlert, dismiss: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(Color(0xFF111111)).padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Box(Modifier.size(36.dp).clip(RoundedCornerShape(10.dp)).background(Color(0xFF26282B)),
            contentAlignment = Alignment.Center) {
            Icon(R.drawable.alert, Modifier.size(18.dp), ServerGreen)
        }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Label(alert.title, 13, Color.White, FontWeight.Bold)
            Label(alert.posture.advice, 12, FeedbackMuted)
        }
        Box(Modifier.size(24.dp).semantics { contentDescription = "자세 알림 닫기" }
            .clickable(role = Role.Button, onClick = dismiss), contentAlignment = Alignment.Center) {
            Icon(R.drawable.x, Modifier.size(18.dp), FeedbackMuted)
        }
    }
}

private fun JSONObject.upperBodyResult(kind: String): String = liveUpperBodyLabel(
    optString("${kind}_state"), optBoolean("pose_detected"), optJSONObject("confidence")?.number(kind)
)

private fun JSONObject.pendingPosture(kind: String): JSONObject? =
    optJSONArray("pending").records().firstOrNull { it.optString("kind") == kind }

private fun JSONObject.remainingTime(): String? = number("remaining_sec")?.let { "${ceil(it).toInt()}초 남음" }

@Composable
private fun LivePostureCards(current: JSONObject, layout: JSONObject) {
    val gap = 12.dp
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        if (maxWidth >= 340.dp && LocalDensity.current.fontScale <= 1.2f) {
            val cardWidth = (maxWidth - gap) / 2
            UpperBodyCard(current, Modifier.matchParentSize().padding(end = cardWidth + gap))
            SeatPressureCard(current, layout, Modifier.width(cardWidth).align(Alignment.TopEnd))
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(gap)) {
                UpperBodyCard(current, Modifier.fillMaxWidth())
                SeatPressureCard(current, layout, Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun UpperBodyCard(current: JSONObject, modifier: Modifier) {
    val head = current.upperBodyResult("head")
    val tilt = current.upperBodyResult("tilt")
    val distanceLevel = postureName(current.optString("distance_level"), LivePostureWaiting)
    val distanceCaution = distanceLevel == "거리 주의" || distanceLevel == "너무 가까워요"
    val tilted = tilt != LivePostureWaiting && tilt != "기울기 없음"
    val status = when {
        head == "거북목 자세" || tilted || distanceCaution -> "주의"
        head == "정상 자세" && tilt == "기울기 없음" -> "바른 자세"
        else -> LivePostureWaiting
    }

    DesignCard(modifier.testTag("upper-body-card"), padding = 16, cornerRadius = 20) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            PostureCardSource(R.drawable.home_posture_camera, "거리센서 · 카메라")
            PostureCardStatus(status)
            PostureGauges(
                headState = current.optString("head_state").takeIf { head != LivePostureWaiting },
                tiltState = current.optString("tilt_state").takeIf { tilt != LivePostureWaiting },
                distanceMm = current.number("distance_mm"),
                distanceLevel = distanceLevel,
                distanceCaution = distanceCaution,
                headCountdown = current.pendingPosture("head")?.remainingTime(),
                tiltCountdown = current.pendingPosture("tilt")?.remainingTime()
            )
            BoxWithConstraints(Modifier.fillMaxWidth()) {
                val caption = if (maxWidth >= (146 * LocalDensity.current.fontScale).dp) {
                    "처음 측정한 기준 자세와 비교해요"
                } else "기준 자세와 비교해요"
                Label(caption, 10, FeedbackMuted)
            }
        }
    }
}

private data class PressureReading(val x: Double, val y: Double, val percent: Int?, val raw: Int?)

@Composable
private fun SeatPressureCard(current: JSONObject, layout: JSONObject, modifier: Modifier) {
    val positions = layout.optJSONArray("positions")
    val ratios = current.optJSONArray("pressure_ratio")
    val pressure = current.optJSONArray("pressure")
    val readings = if (positions == null) emptyList() else (0 until positions.length()).mapNotNull { index ->
        val point = positions.optJSONArray(index) ?: return@mapNotNull null
        val x = point.number(0) ?: return@mapNotNull null
        val y = point.number(1) ?: return@mapNotNull null
        PressureReading(x, y, ratios?.number(index)?.let { (it * 100).roundToInt() },
            pressure?.number(index)?.toInt())
    }
    val rows: List<List<PressureReading?>> = readings.groupBy { it.y }.toSortedMap(reverseOrder())
        .values.map { row -> row.sortedBy { it.x } }
        .ifEmpty { List(PostureRules.PRESSURE_SENSOR_COUNT / 2) { listOf(null, null) } }

    val state = current.optString("seat_state")
    val status = when (state) {
        "normal" -> "정상"
        "lean_left" -> "왼쪽 편향"
        "lean_right" -> "오른쪽 편향"
        else -> seatKindLabel(state, LivePostureWaiting).takeUnless { it == LivePostureWaiting }
            ?: seatSideLabel(current.optString("seat_side")) ?: LivePostureWaiting
    }
    DesignCard(modifier.testTag("seat-pressure-card"), padding = 16, cornerRadius = 20) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            PostureCardSource(R.drawable.seat, "좌면 · 압력센서")
            PostureCardStatus(status)
            Column(Modifier.align(Alignment.CenterHorizontally).widthIn(max = 125.dp).fillMaxWidth()
                .heightIn(min = 122.dp), verticalArrangement = Arrangement.spacedBy(4.dp, Alignment.CenterVertically)) {
                rows.forEach { row ->
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        row.forEach { reading ->
                            val readingLabel = reading?.percent?.let { "$it%" } ?: reading?.raw?.toString() ?: "—"
                            val rawValue = reading?.raw
                            val cell = if (rawValue == null) Modifier else Modifier.semantics {
                                contentDescription = "센서 원시값 $rawValue"
                            }
                            PressureCell(reading?.percent, Modifier.weight(1f).then(cell), readingLabel)
                        }
                    }
                }
            }
            Column {
                val countdown = current.pendingPosture("seat")?.remainingTime()
                val sensorCount = rows.sumOf { it.size }
                Label("앞·뒤 / 좌·우 압력 비율", 10, ServerMuted)
                Label(if (countdown == null) "FSR ${sensorCount}개 기준" else "FSR ${sensorCount}개 · $countdown", 10, ServerMuted)
            }
        }
    }
}

@Composable
private fun PostureCardSource(icon: Int, title: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Icon(icon, Modifier.size(15.dp), ServerMuted)
        Label(title, 10, ServerMuted, FontWeight.Medium)
    }
}

@Composable
private fun PostureCardStatus(status: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        val dot = when (status) {
            "정상", "바른 자세" -> ServerGreen
            LivePostureWaiting -> FeedbackMuted
            "주의" -> Color(0xFF111111)
            else -> Color(0xFF374151)
        }
        Box(Modifier.size(8.dp).clip(CircleShape).background(dot))
        Label(status, 18, weight = FontWeight.Bold)
    }
}

@Composable
internal fun ServerReport(
    report: JSONObject,
    error: String,
    weekly: Boolean,
    setWeekly: (Boolean) -> Unit,
    date: Long,
    selectDate: (Long) -> Unit
) {
    var calendar by rememberSaveable { mutableStateOf(false) }
    if (calendar) ReportCalendar(date, weekly, { calendar = false }, selectDate)
    Column(Modifier.fillMaxSize()) {
        ReportControls(weekly, setWeekly, date, selectDate) { calendar = true }
        HorizontalDivider(color = ChartTrack)
        key(weekly, date) {
            ServerColumn(Modifier.weight(1f)) {
                if (error.isNotBlank()) Label(error, 13, MaterialTheme.colorScheme.error)
                if (report.length() == 0) Label("기록을 기다리고 있어요", 14, ServerMuted)
                else if (weekly) WeeklyServerReport(report) else DailyServerReport(report)
            }
        }
    }
}

@Composable
private fun ReportControls(
    weekly: Boolean,
    setWeekly: (Boolean) -> Unit,
    date: Long,
    selectDate: (Long) -> Unit,
    openCalendar: () -> Unit
) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 20.dp).padding(top = 4.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Label("리포트", 24, weight = FontWeight.Bold, modifier = Modifier.weight(1f))
            IconButton(onClick = openCalendar, modifier = Modifier.semantics { contentDescription = "리포트 날짜 선택" }) {
                Icon(R.drawable.calendar, Modifier.size(20.dp))
            }
        }
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(13.dp)).background(ChartTrack).padding(4.dp).selectableGroup()) {
            listOf(false to "일간", true to "주간").forEach { (mode, title) ->
                val selected = weekly == mode
                Box(
                    Modifier.weight(1f).heightIn(min = 40.dp).clip(RoundedCornerShape(10.dp))
                        .background(if (selected) Color.White else Color.Transparent)
                        .selectable(selected, role = Role.Tab) { setWeekly(mode) }.padding(horizontal = 6.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Label(title, 14, if (selected) MaterialTheme.colorScheme.onSurface else ServerMuted,
                        if (selected) FontWeight.Bold else FontWeight.Normal)
                }
            }
        }
        val offset = if (weekly) 7 else 1
        val next = ReportDates.shift(date, offset)
        val canGoForward = next <= ReportDates.today()
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = { selectDate(ReportDates.shift(date, -offset)) },
                modifier = Modifier.semantics { contentDescription = "이전 기간" }) {
                Icon(R.drawable.chevl, Modifier.size(18.dp))
            }
            TextButton(onClick = openCalendar, modifier = Modifier.weight(1f)) {
                Label(ReportDates.label(date, weekly), 13, weight = FontWeight.Bold, align = TextAlign.Center)
            }
            IconButton(enabled = canGoForward, onClick = { selectDate(next) },
                modifier = Modifier.semantics { contentDescription = "다음 기간" }) {
                Icon(R.drawable.chevr, Modifier.size(18.dp), if (canGoForward) ServerMuted else ChartTrack)
            }
        }
    }
}

@Composable
private fun DailyServerReport(report: JSONObject) {
    val seated = report.number("seated_sec")
    val balance = report.optJSONObject("lean_balance")
    val distance = report.optJSONObject("distance")
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            MetricCard("총 착석시간", seconds(seated), Modifier.weight(1f))
            MetricCard("정상 자세 비율", percent(report.number("normal_ratio")), Modifier.weight(1f), ServerGreen)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            MetricCard("최대 연속 착석", seconds(report.number("max_continuous_sec")), Modifier.weight(1f))
            MetricCard("자세 무너짐 시작", seconds(report.optJSONObject("collapse")?.number("avg_sec")), Modifier.weight(1f))
        }
    }
    val postures = report.optJSONObject("postures")
    DesignCard(Modifier.fillMaxWidth()) {
        Label("자세별 누적시간", 16, weight = FontWeight.Bold)
        if (postures == null || postures.length() == 0) Label("기록 없음", 13, ServerMuted)
        postures?.keys()?.forEach { key ->
            val duration = postures.number(key)
            Spacer(Modifier.height(8.dp))
            DataRow(postureName(key), seconds(duration))
            if (duration != null && seated != null && seated > 0) {
                RatioBar((duration / seated).toFloat().coerceIn(0f, 1f), if (key == "normal") ServerGreen else ServerMuted)
            }
        }
    }
    DesignCard(Modifier.fillMaxWidth()) {
        Label("좌우 체중 편향 비교 (추정)", 16, weight = FontWeight.Bold)
        val left = balance?.number("left_sec")
        val right = balance?.number("right_sec")
        DataRow("왼쪽", seconds(left))
        DataRow("오른쪽", seconds(right))
        if (left != null && right != null && left + right > 0) {
            RatioBar((left / (left + right)).toFloat().coerceIn(0f, 1f), ServerGreen)
        }
    }
    DesignCard(Modifier.fillMaxWidth()) {
        Label("화면과의 거리", 16, weight = FontWeight.Bold)
        DataRow("평균 거리", distanceLabel(distance?.number("avg_mm")) ?: "기록 없음")
        DataRow("기준보다 10cm 이상 가까웠던 시간", seconds(distance?.number("closer_sec")))
    }
}

@Composable
private fun WeeklyServerReport(report: JSONObject) {
    DesignCard(Modifier.fillMaxWidth()) {
        val ratio = report.number("normal_ratio")
        val delta = report.number("delta_pp")
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Label("주간 정상 자세 비율", 14, ServerMuted)
                Label(percent(ratio), if (ratio == null) 18 else 36, weight = FontWeight.Bold)
                if (delta != null) Pill(String.format(Locale.KOREA, "지난주 대비 %+.1f%%p", delta), green = delta >= 0)
                else Label("비교 기록 없음", 12, ServerMuted)
            }
            if (ratio != null) ProgressRing((ratio * 100).toFloat().coerceIn(0f, 100f), Modifier.size(74.dp))
        }
    }
    val days = report.optJSONArray("days").records()
    DesignCard(Modifier.fillMaxWidth()) {
        Label("요일별 착석시간", 16, weight = FontWeight.Bold)
        BarChart(
            days.map { day ->
                val date = day.optString("date")
                val weekday = listOf("월", "화", "수", "목", "금", "토", "일").getOrNull(day.optInt("weekday", -1))
                val total = day.number("seated_sec")?.takeIf { day.optBoolean("has_data", it > 0) }
                val normal = day.number("normal_sec").takeIf { total != null }
                ChartDay(weekday ?: date.takeLast(2), total, normal, "$date 착석 ${seconds(total)}, 정상 ${seconds(normal)}")
            },
            normalAndOther = true
        )
        if (days.any { it.optBoolean("has_data") }) {
            Label("초록색 · 정상 자세   회색 · 그 외 자세", 11, ServerMuted)
        }
    }
    val compare = report.optJSONObject("compare")
    DesignCard(Modifier.fillMaxWidth()) {
        Label("지난주와 비교", 16, weight = FontWeight.Bold)
        Label("기록이 있는 날의 하루 평균", 12, ServerMuted)
        val metrics = listOf(
            "forward_head" to "거북목", "cross" to "다리 꼬기 (추정)", "tilt_direction" to "카메라 기울기",
            "max_continuous" to "최대 연속 착석", "distance_mm" to "화면 거리"
        )
        metrics.forEach { (key, title) ->
            val value = compare?.optJSONObject(key)
            val previous = value?.number("last_week")
            val now = value?.number("this_week")
            val previousText = if (key == "distance_mm") distanceLabel(previous) ?: "기록 없음" else seconds(previous)
            val nowText = if (key == "distance_mm") distanceLabel(now) ?: "기록 없음" else seconds(now)
            DataRow(title, "$previousText → $nowText")
        }
    }
    report.number("worst_hour")?.let { hour ->
        DesignCard(Modifier.fillMaxWidth()) {
            Label("이번 주 패턴", 16, weight = FontWeight.Bold)
            Label("${hour.toInt()}시대의 비정상 자세 비율이 가장 높았어요", 14)
        }
    }
}

@Composable
internal fun ServerGoals(report: JSONObject, error: String, selected: Set<Int>, edit: () -> Unit) {
    val goals = report.optJSONArray("goals").records().associateBy { it.optString("goal") }
    val habits = FeedbackHabit.entries.associateBy { it.key }

    Column(
        Modifier.fillMaxSize().background(Color(0xFFF5F6F5)).verticalScroll(rememberScrollState())
            .padding(start = 20.dp, top = 8.dp, end = 20.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Label("피드백", 24, weight = FontWeight.Bold)
        if (error.isNotBlank()) Label(error, 13, MaterialTheme.colorScheme.error)

        FeedbackSummary(goals)
        Row(Modifier.fillMaxWidth().padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            Label("내 목표", 17, weight = FontWeight.Bold, modifier = Modifier.weight(1f))
            Label("편집", 12, ServerGreen, FontWeight.Medium, Modifier.clickable(role = Role.Button, onClick = edit))
        }
        if (selected.isEmpty()) Label("선택한 목표가 없어요", 14, ServerMuted)
        GoalKeys.filterIndexed { index, _ -> index in selected }.forEach { key ->
            FeedbackGoalCard(habits.getValue(key), goals[key])
        }

        Row(Modifier.fillMaxWidth().padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            Label("오늘의 추천", 17, weight = FontWeight.Bold, modifier = Modifier.weight(1f))
            Label("누적 기록 기반", 12, FeedbackMuted, FontWeight.Medium)
        }
        DesignCard(Modifier.fillMaxWidth(), padding = 0) {
            Column(Modifier.padding(horizontal = 18.dp, vertical = 6.dp)) {
                FeedbackHabit.entries.forEachIndexed { index, habit ->
                    if (index > 0) HorizontalDivider(color = Color(0xFFECEDEF))
                    Row(Modifier.fillMaxWidth().padding(vertical = 13.dp),
                        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        FeedbackIcon(habit.activityIcon, Color(0xFFF3F4F6), Color(0xFF374151))
                        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(1.dp)) {
                            Label(habit.activity, 14, weight = FontWeight.Bold)
                            Label(habit.evidence(goals[habit.key]), 12, ServerMuted)
                        }
                        FeedbackPill(habit.duration)
                    }
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Image(painterResource(R.drawable.feedback_info), null, Modifier.size(14.dp))
            Label("생활습관 참고용 안내이며, 의료적 진단이나 치료를 대신하지 않아요.", 11, FeedbackMuted)
        }
    }
}

private val FeedbackMuted = Color(0xFF9CA3AF)

private enum class FeedbackHabit(
    val key: String,
    val title: String,
    val detail: String,
    val activity: String,
    val duration: String,
    val activityIcon: Int,
    val tip: String
) {
    NECK("forward_head", "거북목 자세", "상체 · 카메라 기준", "목·어깨 가볍게 돌리기", "1분",
        R.drawable.stretch, "목을 좌우로 15초씩 돌려주세요."),
    SITTING("long_sitting", "장시간 연속 착석", "착석 시간 기준", "자리에서 일어나 걷기", "2분",
        R.drawable.walk, ""),
    LEAN("lean", "한쪽으로 기대기", "좌면 · 체중 편향 추정", "좌우 균형 확인하기", "10초",
        R.drawable.balance, "양쪽에 체중을 고르게 실어주세요."),
    CROSS("cross", "다리 꼬기", "다리 꼬기 · 방석 압력 기준", "다리 풀고 양발 바닥에 두기", "수시로",
        R.drawable.legs, "양발을 바닥에 편하게 내려놓으세요.");

    val goalIcon: Int get() = goalIcons[GoalKeys.indexOf(key)]
}

private fun feedbackTime(seconds: Double?): String {
    val total = seconds?.coerceAtLeast(0.0)?.toLong() ?: return "—"
    val minutes = total / 60
    return when {
        minutes >= 60 && minutes % 60 == 0L -> "${minutes / 60}시간"
        minutes >= 60 -> "${minutes / 60}시간 ${minutes % 60}분"
        minutes > 0 -> "${minutes}분"
        else -> "${total}초"
    }
}

private fun JSONObject?.upperLegLabel(): String? = when (this?.optJSONObject("main_direction")?.optString("direction")) {
    "left" -> "주로 왼다리 위"
    "right" -> "주로 오른다리 위"
    else -> null
}

private fun FeedbackHabit.evidence(goal: JSONObject?): String {
    if (this == FeedbackHabit.LEAN) {
        val direction = goal?.optJSONObject("main_direction")
        val left = direction?.number("left_sec") ?: 0.0
        val right = direction?.number("right_sec") ?: 0.0
        if (left > 0 || right > 0) {
            val longer = maxOf(left, right)
            val shorter = minOf(left, right)
            val side = if (left > right) "왼쪽" else "오른쪽"
            return when {
                shorter == 0.0 -> "${side}으로 기댄 시간이 ${feedbackTime(longer)} 있었어요"
                longer / shorter < 1.05 -> "좌우로 기댄 시간이 비슷했어요"
                else -> "${side}으로 기댄 시간이 ${String.format(Locale.KOREA, "%.1f", longer / shorter)}배 길었어요"
            }
        }
    }
    val today = goal?.number("today") ?: return "기록 없음"
    val durationText = feedbackTime(today)
    return when (this) {
        FeedbackHabit.NECK -> "거북목 자세가 ${durationText} 나타났어요"
        FeedbackHabit.SITTING -> "최대 연속 착석이 $durationText${if (durationText.endsWith("초")) "였어요" else "이었어요"}"
        FeedbackHabit.LEAN -> "한쪽으로 기댄 시간이 ${durationText} 있었어요"
        FeedbackHabit.CROSS -> "다리 꼬기 $durationText" + goal.upperLegLabel()?.let { " · $it" }.orEmpty()
    }
}

@Composable
private fun FeedbackSummary(goals: Map<String, JSONObject>) {
    val most = FeedbackHabit.entries.filter { it != FeedbackHabit.SITTING }
        .mapNotNull { habit -> goals[habit.key]?.number("today")?.takeIf { it > 0 }?.let { habit to it } }
        .maxByOrNull { it.second }

    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(24.dp)).background(Color(0xFF111111)).padding(20.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Label("오늘 가장 많이 나타난 습관", 12, FeedbackMuted, FontWeight.Medium)
            Label(most?.first?.title ?: "기록 없음", 22, Color.White, FontWeight.Bold)
            if (most != null) {
                Label(feedbackTime(most.second), 25, ServerGreen, FontWeight.Bold)
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Image(painterResource(R.drawable.feedback_tip_bulb), null, Modifier.size(12.dp, 15.dp))
                    Label(most.first.tip, 12, FeedbackMuted)
                }
            }
        }
        Box(Modifier.size(56.dp).clip(RoundedCornerShape(16.dp)).background(Color(0xFF26282B)),
            contentAlignment = Alignment.Center) {
            Icon(most?.first?.goalIcon ?: R.drawable.bulb, Modifier.size(26.dp), ServerGreen)
        }
    }
}

@Composable
private fun FeedbackGoalCard(habit: FeedbackHabit, goal: JSONObject?) {
    val trend = goal?.optJSONObject("trend")?.optString("direction")
    val trendLabel = when (trend) {
        "decreasing" -> "감소 추세"
        "increasing" -> "증가 추세"
        "flat" -> "비슷한 추세"
        else -> null
    }
    DesignCard(Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            FeedbackIcon(habit.goalIcon, ChartMint, ServerGreen)
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(1.dp)) {
                Label("${habit.title} 줄이기", 15, weight = FontWeight.Bold)
                val detail = if (habit == FeedbackHabit.CROSS) goal.upperLegLabel()?.let { "다리 꼬기 · $it" } else null
                Label(detail ?: habit.detail, 12, ServerMuted)
            }
            trendLabel?.let { FeedbackPill(it, green = trend == "decreasing") }
        }
        Spacer(Modifier.height(14.dp))
        val days = goal?.optJSONArray("last7").records().takeLast(7)
        BoxWithConstraints(Modifier.fillMaxWidth()) {
            if (maxWidth >= 270.dp && LocalDensity.current.fontScale <= 1.2f) {
                Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    FeedbackMetrics(goal, Modifier.weight(1f))
                    if (days.isNotEmpty()) FeedbackMiniBars(days)
                }
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    FeedbackMetrics(goal, Modifier.fillMaxWidth())
                    if (days.isNotEmpty()) FeedbackMiniBars(days, Modifier.align(Alignment.End))
                }
            }
        }
    }
}

@Composable
private fun FeedbackMetrics(goal: JSONObject?, modifier: Modifier) {
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        listOf("오늘" to "today", "어제" to "yesterday", "지난주 평균" to "last_week_avg").forEach { (title, field) ->
            val today = field == "today"
            Column(Modifier.weight(1f, fill = false), verticalArrangement = Arrangement.spacedBy(1.dp)) {
                Label(title, 11, ServerMuted, FontWeight.Medium)
                Label(feedbackTime(goal?.number(field)), if (today) 19 else 16,
                    if (today) ServerGreen else Color(0xFF111111), FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun FeedbackMiniBars(days: List<JSONObject>, modifier: Modifier = Modifier) {
    val values = days.map { it.number("value") }
    val peak = values.filterNotNull().maxOrNull()?.coerceAtLeast(1.0) ?: 1.0
    val description = days.joinToString(", ") { day -> "${day.optString("date")} ${feedbackTime(day.number("value"))}" }
    Row(modifier.height(40.dp).clearAndSetSemantics { contentDescription = description },
        verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        values.forEachIndexed { index, value ->
            val height = (((value ?: 0.0) / peak) * 40).dp.coerceAtLeast(2.dp)
            val color = if (index == values.lastIndex && value != null) ServerGreen else Color(0xFFDDE0E4)
            Box(Modifier.width(7.dp).height(height).clip(RoundedCornerShape(3.dp)).background(color))
        }
    }
}

@Composable
private fun FeedbackIcon(icon: Int, background: Color, tint: Color) {
    Box(Modifier.size(40.dp).clip(RoundedCornerShape(12.dp)).background(background), contentAlignment = Alignment.Center) {
        Icon(icon, Modifier.size(20.dp), tint)
    }
}

@Composable
private fun FeedbackPill(text: String, green: Boolean = true) {
    Label(text, 11, if (green) ServerGreen else ServerMuted, FontWeight.Medium,
        Modifier.clip(CircleShape).background(if (green) ChartMint else Color(0xFFF3F4F6))
            .padding(horizontal = 10.dp, vertical = 4.dp))
}

private fun goalResultLabel(goal: JSONObject): String {
    val label = goal.optString("label")
    return when (goal.optString("goal")) {
        "cross", "lean" -> "$label (추정)"
        else -> label
    }
}

@Composable
private fun GoalIcon(goal: String) {
    val icon = goalIcons.getOrNull(GoalKeys.indexOf(goal)) ?: R.drawable.target
    Box(Modifier.size(38.dp).clip(RoundedCornerShape(12.dp)).background(ChartMint), contentAlignment = Alignment.Center) {
        Icon(icon, Modifier.size(20.dp), ServerGreen)
    }
}

@Composable
private fun DataRow(title: String, value: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Label(title, 13, ServerMuted, modifier = Modifier.weight(1f))
        Label(value, 13, weight = FontWeight.SemiBold, modifier = Modifier.weight(1f), align = TextAlign.End)
    }
}

private data class ChartDay(val label: String, val total: Double?, val normal: Double?, val description: String)

@Composable
private fun BarChart(days: List<ChartDay>, normalAndOther: Boolean = false) {
    if (days.none { it.total != null }) {
        Label("기록 없음", 12, ServerMuted, modifier = Modifier.padding(top = 6.dp))
        return
    }
    val maxValue = days.maxOf { it.total ?: 0.0 }.coerceAtLeast(1.0)
    Row(Modifier.fillMaxWidth().padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        days.forEach { day ->
            Column(Modifier.weight(1f).clearAndSetSemantics { contentDescription = day.description },
                horizontalAlignment = Alignment.CenterHorizontally) {
                Box(Modifier.fillMaxWidth().height(68.dp), contentAlignment = Alignment.BottomCenter) {
                    val total = day.total ?: 0.0
                    if (total > 0) {
                        val height = (total / maxValue * 64).dp.coerceAtLeast(3.dp)
                        val normalFraction = if (normalAndOther) ((day.normal ?: 0.0) / total).coerceIn(0.0, 1.0) else 1.0
                        Box(Modifier.widthIn(max = 24.dp).fillMaxWidth().height(height)
                            .clip(RoundedCornerShape(4.dp)).background(ChartTrack), contentAlignment = Alignment.BottomCenter) {
                            Box(Modifier.fillMaxWidth().fillMaxHeight(normalFraction.toFloat()).background(ServerGreen))
                        }
                    } else if (day.total != null) {
                        Box(Modifier.widthIn(max = 24.dp).fillMaxWidth().height(2.dp).background(ServerGreen))
                    }
                }
                Spacer(Modifier.height(3.dp))
                Label(day.label, 10, ServerMuted, modifier = Modifier.fillMaxWidth(), align = TextAlign.Center)
                if (day.total == null) {
                    Label("—", 10, ServerMuted)
                } else {
                    val minutes = (day.total / 60).toInt()
                    val display = when {
                        minutes >= 60 -> String.format(Locale.KOREA, "%.1f시간", day.total / 3600)
                        minutes > 0 -> "${minutes}분"
                        else -> "${day.total.toInt()}초"
                    }
                    Label(display, 10, ServerMuted, modifier = Modifier.fillMaxWidth(), align = TextAlign.Center)
                }
            }
        }
    }
}

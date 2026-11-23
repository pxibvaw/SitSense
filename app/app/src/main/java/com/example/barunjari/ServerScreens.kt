package com.example.barunjari

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
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
        val pending = current.optJSONArray("pending")
        if (pending != null && pending.length() > 0) DesignCard(Modifier.fillMaxWidth()) {
            Label("판정 대기 중", 17, ServerGreen, FontWeight.Bold)
            Label("자세가 유지되는지 확인하고 있어요. 반영까지 약 5초 걸릴 수 있어요.", 12, ServerMuted)
            repeat(pending.length()) { i ->
                val item = pending.optJSONObject(i) ?: return@repeat
                val state = item.optString("state")
                val posture = when (item.optString("kind")) {
                    "seat" -> "좌면 · ${seatKindLabel(state)}"
                    "head" -> "목 · ${postureName(state)}"
                    else -> "몸 기울기 · ${postureName(state)}"
                }
                Label("$posture · ${ceil(item.optDouble("remaining_sec", 0.0)).toInt()}초 남음", 13)
                val progress by androidx.compose.animation.core.animateFloatAsState(
                    targetValue = (1f - item.optDouble("remaining_sec", 4.0).toFloat() / 4f).coerceIn(0f, 1f),
                    animationSpec = androidx.compose.animation.core.tween(1000), label = "pending")
                RatioBar(progress, ServerGreen)
            }
        }
        Label("지금 자세", 17, weight = FontWeight.Bold)
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

private data class PressureReading(val x: Double, val y: Double, val percent: Int?)

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun LivePostureCards(current: JSONObject, layout: JSONObject) {
    val positions = layout.optJSONArray("positions")
    val ratios = current.optJSONArray("pressure_ratio")
    val readings = if (positions == null) emptyList() else (0 until positions.length()).mapNotNull { index ->
        val point = positions.optJSONArray(index) ?: return@mapNotNull null
        val x = point.number(0) ?: return@mapNotNull null
        val y = point.number(1) ?: return@mapNotNull null
        PressureReading(x, y, ratios?.number(index)?.let { (it * 100).roundToInt() })
    }
    val rows = readings.groupBy { it.y }.toSortedMap(reverseOrder())
    val poseDetected = current.optBoolean("pose_detected")
    val confidence = current.optJSONObject("confidence")
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val columns = if (maxWidth >= 340.dp && LocalDensity.current.fontScale <= 1.2f) 2 else 1
        FlowRow(Modifier.fillMaxWidth(), maxItemsInEachRow = columns,
            horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            DesignCard(Modifier.weight(1f), padding = 14) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Icon(R.drawable.camera, Modifier.size(18.dp), ServerMuted)
                    Label("상체 · 화면 거리", 14, weight = FontWeight.Bold)
                }
                Spacer(Modifier.height(8.dp))
                val head = liveUpperBodyLabel(current.optString("head_state"), poseDetected, confidence?.number("head"))
                val tilt = liveUpperBodyLabel(current.optString("tilt_state"), poseDetected, confidence?.number("tilt"))
                Label("목 · $head", 13)
                Label("몸 · $tilt", 13)
                Spacer(Modifier.height(8.dp))
                Label("화면 거리", 12, ServerMuted)
                val distance = distanceLabel(current.number("distance_mm"))
                Label(distance ?: "측정 대기 중", 16, weight = FontWeight.Bold)
                if (distance != null) Label(postureName(current.optString("distance_level"), LivePostureWaiting), 12, ServerMuted)
            }
            DesignCard(Modifier.weight(1f), padding = 14) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Icon(R.drawable.seat, Modifier.size(18.dp), ServerMuted)
                    Label("좌면 · 방석 압력", 14, weight = FontWeight.Bold)
                }
                Spacer(Modifier.height(8.dp))
                seatSideLabel(current.optString("seat_side"))?.let { Label(it, 13, weight = FontWeight.SemiBold) }
                Label(seatKindLabel(current.optString("seat_state"), LivePostureWaiting), 13)
                if (rows.isNotEmpty()) {
                    Spacer(Modifier.height(8.dp))
                    Label("앞 · 왼쪽에서 오른쪽 순", 10, ServerMuted)
                    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        rows.values.forEach { row ->
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                row.sortedBy { it.x }.forEach { reading -> PressureCell(reading.percent, Modifier.weight(1f)) }
                            }
                        }
                    }
                    Label("뒤", 10, ServerMuted)
                } else Label("센서 배치를 불러오지 못했어요", 12, ServerMuted)
            }
        }
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
    ServerColumn {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Label("피드백", 24, weight = FontWeight.Bold, modifier = Modifier.weight(1f))
            TextButton(onClick = edit) { Text("목표 수정") }
        }
        if (error.isNotBlank()) Label(error, 13, MaterialTheme.colorScheme.error)
        val goals = report.optJSONArray("goals").records()
        if (selected.isEmpty()) Label("선택한 목표가 없어요", 14, ServerMuted)
        else if (goals.isEmpty()) Label("목표별 기록을 기다리고 있어요", 14, ServerMuted)
        goals.filter { GoalKeys.indexOf(it.optString("goal")) in selected }.forEach { goal ->
            val key = goal.optString("goal")
            DesignCard(Modifier.fillMaxWidth()) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    GoalIcon(key)
                    Label(goalResultLabel(goal), 17, weight = FontWeight.Bold)
                }
                Spacer(Modifier.height(12.dp))
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("오늘" to "today", "어제" to "yesterday", "지난주 평균" to "last_week_avg").forEach { (title, field) ->
                        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                            Label(title, 11, ServerMuted)
                            Label(seconds(goal.number(field)), 13,
                                color = if (field == "today") ServerGreen else MaterialTheme.colorScheme.onSurface,
                                weight = FontWeight.Bold)
                        }
                    }
                }
                val days = goal.optJSONArray("last7").records()
                if (days.isNotEmpty()) {
                    Spacer(Modifier.height(12.dp))
                    Label("최근 7일", 12, ServerMuted)
                    BarChart(days.map { day ->
                        val date = day.optString("date")
                        val value = day.number("value")
                        ChartDay(date.takeLast(2), value, null, "$date ${seconds(value)}")
                    })
                }
                Spacer(Modifier.height(8.dp))
                val tip = when (key) {
                    "cross" -> "두 발을 바닥에 붙이고 다리를 풀어 주세요"
                    "forward_head" -> "턱을 살짝 당기고 화면을 정면으로 봐 주세요"
                    "lean" -> "양쪽 좌면에 체중을 고르게 실어 주세요"
                    else -> "잠시 일어나 가볍게 걸어 주세요"
                }
                Label(tip, 13, modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp))
                    .background(ChartMint).padding(10.dp))
            }
        }
    }
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

package com.example.barunjari

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.requiredWidth
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.getUnclippedBoundsInRoot
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class LivePostureScreenTest {
    @get:Rule val compose = createComposeRule()

    @Test fun homeShowsTheMatchingAlertAndDismissesIt() {
        val alert = mutableStateOf<PostureAlert?>(PostureAlert(AlertPosture.NECK, 1))
        compose.setContent {
            ServerHome(JSONObject(), JSONObject(), JSONObject(), JSONObject(), "", postureAlert = alert.value,
                dismissPostureAlert = { alert.value = null }) {}
        }
        compose.onNodeWithText("거북목 자세가 1분째 이어지고 있어요").assertExists()
        compose.onNodeWithContentDescription("자세 알림 닫기").performScrollTo().performClick()
        compose.onNodeWithText("거북목 자세가 1분째 이어지고 있어요").assertDoesNotExist()
    }

    @Test fun postureCardsStayVisibleWhenPendingAndCurrentDataClear() {
        val confirmed = """{
            "pose_detected":true,"head_state":"normal","tilt_state":"right",
            "seat_state":"cross_left","seat_side":"right","sit_position":"back",
            "distance_level":"caution","distance_mm":456
        }"""
        val current = mutableStateOf(JSONObject(confirmed).put(
            "pending", JSONArray("""[{"kind":"head","state":"forward","remaining_sec":2}]""")
        ))
        compose.setContent { ServerHome(JSONObject(), current.value, JSONObject(), JSONObject(), "") {} }

        compose.onNodeWithText("지금 자세").assertExists()
        compose.onNodeWithText("목 · 2초 남음").assertExists()
        compose.onNodeWithText("다리 꼬기 (추정)").assertExists()
        compose.onNodeWithText("주의").assertExists()
        compose.onNodeWithText("몸 오른쪽").assertExists()
        compose.onNodeWithText("화면 거리 45.6cm").assertExists()
        compose.onNodeWithContentDescription("거리 주의").assertExists()
        compose.onNodeWithText("체중 방향").assertDoesNotExist()
        compose.onNodeWithText("앉은 위치").assertDoesNotExist()

        compose.runOnIdle { current.value = JSONObject(confirmed) }
        compose.onNodeWithText("목 정상").assertExists()
        compose.onNodeWithText("2초 남음", substring = true).assertDoesNotExist()

        compose.runOnIdle { current.value = JSONObject(confirmed).put("seat_state", "lean_right") }
        compose.onNodeWithText("오른쪽 편향").assertExists()

        compose.runOnIdle { current.value = JSONObject() }
        compose.onNodeWithText("거리센서 · 카메라").assertExists()
        compose.onNodeWithText("좌면 · 압력센서").assertExists()
        compose.onAllNodesWithText(LivePostureWaiting).assertCountEquals(2)
        compose.onAllNodesWithText("—").assertCountEquals(6)
    }

    @Test fun pressureCellsShowPercentagesAndTheCardsHaveTheSameSize() {
        val current = mutableStateOf(JSONObject(
            """{"pressure":[789,123,4095,0,null,2048],"pressure_ratio":[0.2,0.1,0.4,null,null,0.3]}"""
        ))
        val layout = JSONObject("""{"positions":[[1,0],[-1,1],[1,-1],[1,1],[-1,-1],[-1,0]]}""")
        compose.setContent {
            CompositionLocalProvider(LocalDensity provides Density(LocalDensity.current.density, fontScale = 1f)) {
                Box(Modifier.requiredWidth(390.dp)) {
                    ServerHome(JSONObject(), current.value, layout, JSONObject(), "") {}
                }
            }
        }

        val upper = compose.onNodeWithTag("upper-body-card").getUnclippedBoundsInRoot()
        val seat = compose.onNodeWithTag("seat-pressure-card").getUnclippedBoundsInRoot()
        assertEquals(upper.top, seat.top)
        assertEquals((upper.right - upper.left).value, (seat.right - seat.left).value, 1f)
        assertEquals(upper.bottom, seat.bottom)

        val expectedRows = listOf("10%" to "0", "30%" to "20%", "—" to "40%")
        expectedRows.forEach { (left, right) ->
            val leftCell = compose.onNodeWithText(left).getUnclippedBoundsInRoot()
            val rightCell = compose.onNodeWithText(right).getUnclippedBoundsInRoot()
            assertEquals(leftCell.top, rightCell.top)
            assertTrue(leftCell.left < rightCell.left)
        }
        compose.onNodeWithContentDescription("센서 원시값 123").assertExists()

        compose.runOnIdle {
            current.value = JSONObject(current.value.toString()).put("seat_side", "left").put("seat_state", "lean_left")
                .put("pose_detected", true).put("head_state", "forward").put("tilt_state", "right")
                .put("distance_mm", 459).put("distance_level", "caution")
        }
        val updatedSeat = compose.onNodeWithTag("seat-pressure-card").getUnclippedBoundsInRoot()
        val updatedUpper = compose.onNodeWithTag("upper-body-card").getUnclippedBoundsInRoot()
        assertEquals(seat.bottom - seat.top, updatedSeat.bottom - updatedSeat.top)
        assertEquals(updatedSeat.bottom, updatedUpper.bottom)
        compose.onNodeWithTag("seat-pressure-card").performScrollTo()
        compose.captureDesignCheck("home-figma-gauges.png")
    }
}

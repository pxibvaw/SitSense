package com.example.barunjari

import org.junit.Assert.*
import org.junit.Test

class ServerMessagesTest {
    @Test fun pressureFailureExplainsHowToRetry() {
        assertEquals("방석에 깊숙이 바르게 앉아 다시 측정해 주세요", baselineMessage("failed", "pressure_low"))
        assertTrue(baselineMessage("failed", "sensor_lost").contains("센서"))
        assertTrue(baselineMessage("waiting_seat", "").contains("앉아"))
    }
    @Test fun unknownAndFutureStatesNeverAppearAsNormal() {
        assertEquals("판정 불가", postureName("unknown"))
        assertEquals("판정 불가", postureName("future_state"))
        assertEquals("정상 자세", postureName("normal"))
    }

    @Test fun liveCardsShowWaitingWhenPostureIsNotRecognized() {
        assertEquals(LivePostureWaiting, postureName("", LivePostureWaiting))
        assertEquals(LivePostureWaiting, postureName("future_state", LivePostureWaiting))
        assertEquals(LivePostureWaiting, seatKindLabel("unknown", LivePostureWaiting))
        assertEquals("자리 비움", seatKindLabel("empty", LivePostureWaiting))
        assertEquals(LivePostureWaiting, liveUpperBodyLabel("normal", false, 0.8))
        assertEquals(LivePostureWaiting, liveUpperBodyLabel("normal", true, 0.0))
        assertEquals("거북목 자세", liveUpperBodyLabel("forward", true, 0.8))
    }
    @Test fun allBaselineStatesHaveDifferentMessages() {
        assertEquals(5, listOf("waiting_seat", "measuring", "ready", "failed", "none")
            .map { baselineMessage(it, "no_person") }.toSet().size)
    }

    @Test fun distanceIsShownInCentimetersToOneDecimalPlace() {
        assertEquals("47.6cm", distanceLabel(476.05))
        assertEquals("57.7cm", distanceLabel(577.0))
        assertNull(distanceLabel(null))
        assertNull(distanceLabel(Double.NaN))
    }

    @Test fun weightDirectionAndSeatKindHaveIndependentLabels() {
        assertEquals("왼쪽 (확정)", seatSideLabel("left"))
        assertEquals("오른쪽 (확정)", seatSideLabel("right"))
        assertNull(seatSideLabel(""))
        assertNull(seatSideLabel("null"))
        assertNull(seatSideLabel("cross_left"))
        assertEquals("다리 꼬기 (추정)", seatKindLabel("cross_left"))
        assertEquals("체중 편향 (추정)", seatKindLabel("lean_right"))
        assertEquals("자리 비움", seatKindLabel("empty"))
        assertEquals("판정 불가", seatKindLabel("unknown"))
    }
}

package com.example.barunjari

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PostureAlertsTest {
    @Test fun everyGoalBecomesEligibleAtOneMinute() {
        AlertPosture.entries.forEach { posture ->
            val tracker = PostureAlertTracker()
            val signal = mapOf(posture to "confirmed")
            for (second in 0..59) tracker.record(second.toDouble(), signal)
            assertNull(tracker.alert())
            tracker.record(60.0, signal)
            assertEquals(PostureAlert(posture, 1), tracker.alert())
        }
    }

    @Test fun dismissalLastsForOneEpisodeAndOtherPosturesCanStillAlert() {
        val tracker = PostureAlertTracker()
        val sitting = mapOf(AlertPosture.SITTING to "seated")
        val withNeck = sitting + (AlertPosture.NECK to "forward")
        for (second in 0..60) tracker.record(second.toDouble(), withNeck)
        assertEquals(AlertPosture.NECK, tracker.alert()?.posture)
        tracker.dismiss(AlertPosture.NECK)
        assertEquals(AlertPosture.SITTING, tracker.alert()?.posture)
        tracker.dismiss(AlertPosture.SITTING)
        tracker.record(61.0, withNeck)
        assertNull(tracker.alert())

        tracker.record(62.0, sitting)
        for (second in 63..123) tracker.record(second.toDouble(), withNeck)
        assertEquals(PostureAlert(AlertPosture.NECK, 1), tracker.alert())
    }

    @Test fun aDirectionChangeAndADataGapRestartTheTimer() {
        val tracker = PostureAlertTracker()
        val left = mapOf(AlertPosture.LEAN to "lean_left")
        val right = mapOf(AlertPosture.LEAN to "lean_right")
        for (second in 0..59) tracker.record(second.toDouble(), left)
        tracker.record(60.0, right)
        assertNull(tracker.alert())
        for (second in 61..120) tracker.record(second.toDouble(), right)
        assertEquals(AlertPosture.LEAN, tracker.alert()?.posture)
        tracker.record(131.0, right)
        assertNull(tracker.alert())
        tracker.reset()
        assertNull(tracker.alert())
    }
}

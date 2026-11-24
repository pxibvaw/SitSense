package com.example.barunjari

import android.graphics.Bitmap
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.junit4.ComposeContentTestRule
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import java.io.File

class FeedbackScreenTest {
    @get:Rule val compose = createComposeRule()

    @Test fun displaysDesignSectionsWithReportedMeasurements() {
        val report = JSONObject("""{
            "goals":[
                {"goal":"cross","today":1500,"yesterday":2280,"last_week_avg":2280,
                    "main_direction":{"direction":"right","left_sec":1300,"right_sec":2220},
                    "trend":{"direction":"decreasing"},"last7":[{"value":2280},{"value":2100},
                    {"value":2570},{"value":1800},{"value":1750},{"value":2280},{"value":1500}]},
                {"goal":"forward_head","today":2520,"yesterday":3060,"last_week_avg":3480,
                    "trend":{"direction":"decreasing"},"last7":[{"value":3330},{"value":2970},
                    {"value":3600},{"value":2610},{"value":2790},{"value":3060},{"value":2520}]},
                {"goal":"lean","today":450,"main_direction":{"left_sec":600,"right_sec":1680}},
                {"goal":"long_sitting","today":6000}
            ]
        }""")
        var editing = false
        compose.setContent { ServerGoals(report, "", setOf(0, 1)) { editing = true } }

        compose.onNodeWithText("오늘 가장 많이 나타난 습관").assertIsDisplayed()
        compose.onNodeWithText("거북목 자세").assertIsDisplayed()
        compose.onNodeWithText("목을 좌우로 15초씩 돌려주세요.").assertIsDisplayed()
        compose.onNodeWithText("편집").performClick()
        compose.runOnIdle { assertTrue(editing) }
        compose.onNodeWithText("다리 꼬기 줄이기").assertIsDisplayed()
        compose.onNodeWithText("다리 꼬기 · 주로 오른다리 위").assertIsDisplayed()
        compose.onNodeWithText("거북목 자세 줄이기").assertIsDisplayed()
        compose.captureDesignCheck("feedback-figma-top.png")

        compose.onNodeWithText("오늘의 추천").performScrollTo().assertIsDisplayed()
        listOf(
            "거북목 자세가 42분 나타났어요",
            "최대 연속 착석이 1시간 40분이었어요",
            "오른쪽으로 기댄 시간이 2.8배 길었어요",
            "다리 꼬기 25분 · 주로 오른다리 위"
        ).forEach { compose.onNodeWithText(it).performScrollTo().assertIsDisplayed() }
        compose.onNodeWithText("생활습관 참고용 안내이며, 의료적 진단이나 치료를 대신하지 않아요.")
            .performScrollTo().assertIsDisplayed()
        compose.captureDesignCheck("feedback-figma-recommendations.png")
    }

    @Test fun missingMeasurementsKeepTheRecommendationsWithoutExampleNumbers() {
        compose.setContent { ServerGoals(JSONObject(), "", setOf(0, 1)) {} }

        compose.onNodeWithText("오늘 가장 많이 나타난 습관").assertIsDisplayed()
        compose.onNodeWithText("42분").assertDoesNotExist()
        compose.onNodeWithText("거북목 자세가 42분 나타났어요").assertDoesNotExist()
        compose.onNodeWithText("다리 풀고 양발 바닥에 두기").performScrollTo().assertIsDisplayed()
    }

}

internal fun ComposeContentTestRule.captureDesignCheck(name: String) {
    if (InstrumentationRegistry.getArguments().getString("capture_layout") != "true") return
    val image = onRoot().captureToImage().asAndroidBitmap()
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    File(context.cacheDir, name).outputStream().use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
}

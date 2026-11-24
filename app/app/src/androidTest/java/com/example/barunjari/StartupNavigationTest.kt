package com.example.barunjari

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.core.app.ActivityScenario
import org.junit.Rule
import org.junit.Test

class StartupNavigationTest {
    @get:Rule val compose = createEmptyComposeRule()

    @Test fun freshLaunchOpensOnboardingEvenAfterVisitingDeviceSetup() {
        ActivityScenario.launch(MainActivity::class.java).use {
            openDeviceSetup()
        }

        ActivityScenario.launch(MainActivity::class.java).use {
            compose.onNodeWithText("시작하기").assertIsDisplayed()
            compose.onNodeWithText("바른자리").assertDoesNotExist()
        }
    }

    @Test fun deviceSetupBackReturnsThroughGoalsToOnboarding() {
        ActivityScenario.launch(MainActivity::class.java).use {
            openDeviceSetup()
            compose.onNodeWithContentDescription("뒤로").performClick()
            compose.onNodeWithText("어떤 습관을", substring = true).assertIsDisplayed()
            compose.onNodeWithContentDescription("뒤로").performClick()
            compose.onNodeWithText("시작하기").assertIsDisplayed()
        }
    }

    private fun openDeviceSetup() {
        compose.onNodeWithText("시작하기").performClick()
        compose.onNodeWithText("어떤 습관을", substring = true).assertIsDisplayed()
        compose.onNodeWithText("다음").performClick()
        compose.onNodeWithText("기기와 센서를 연결해 주세요").assertIsDisplayed()
    }
}

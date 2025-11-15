package com.symbioza.daymind.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.symbioza.daymind.state.UiState

@Composable
fun DayMindScreen(
    state: UiState,
    onToggleRecording: () -> Unit,
    onRetryUploads: () -> Unit,
    onPlayLastChunk: () -> Unit,
    onStopPlayback: () -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Android Audio Bridge",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )

            Button(onClick = onToggleRecording) {
                Text(if (state.isRecording) "Stop Recording" else "Start Recording")
            }

            Button(
                onClick = { if (state.isPlayingBack) onStopPlayback() else onPlayLastChunk() },
                enabled = state.canPlayChunk || state.isPlayingBack
            ) {
                Text(if (state.isPlayingBack) "Stop Playback" else "Play Last Chunk")
            }

            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(text = "Pending chunks: ${state.pendingChunks}")
                Text(text = "Last upload: ${state.lastUploadMessage}")
            }

            if (state.authError) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "Uploads paused — invalid API key",
                        color = MaterialTheme.colorScheme.error
                    )
                    Button(onClick = onRetryUploads) {
                        Text("Retry uploads")
                    }
                }
            }
        }
    }
}

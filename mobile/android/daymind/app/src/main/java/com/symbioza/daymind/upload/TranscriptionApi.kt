package com.symbioza.daymind.upload

import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface TranscriptionApi {
    @Multipart
    @POST("v1/transcribe")
    suspend fun uploadChunk(
        @Part file: MultipartBody.Part,
        @Part("session_ts") sessionTs: RequestBody,
        @Part("device_id") deviceId: RequestBody,
        @Part("sample_rate") sampleRate: RequestBody,
        @Part("format") format: RequestBody,
        @Part("speech_segments") speechSegments: RequestBody?
    ): Response<ResponseBody>
}

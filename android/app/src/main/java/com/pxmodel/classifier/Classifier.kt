package com.pxmodel.classifier

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class Classifier(context: Context) {

    companion object {
        private const val TAG = "Classifier"
        private const val MODEL_PATH = "efficientnet_b0_multilabel.onnx"
        private const val INPUT_SIZE = 224

        private val LABELS = arrayOf("damaged", "plastic_wrap", "sealed", "open")

        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession

    init {
        val modelBytes = context.assets.open(MODEL_PATH).use { it.readBytes() }
        ortSession = ortEnv.createSession(modelBytes)
        Log.i(TAG, "Model loaded: ${ortSession.inputNames}, ${ortSession.outputNames}")
    }

    fun predict(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        val inputBuffer = preprocess(resized)

        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val tensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)

        val result = ortSession.run(mapOf("input" to tensor))
        val output = result.get("logits") as OnnxTensor
        val logits = output.floatBuffer

        val probabilities = FloatArray(LABELS.size)
        for (i in probabilities.indices) {
            probabilities[i] = sigmoid(logits.get(i))
        }

        tensor.close()
        result.close()

        return probabilities
    }

    private fun preprocess(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val inputChannels = 3
        val floatArray = FloatArray(width * height * inputChannels)
        var idx = 0

        for (row in 0 until height) {
            for (col in 0 until width) {
                val pixel = pixels[row * width + col]
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                floatArray[idx] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                floatArray[idx + width * height] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                floatArray[idx + 2 * width * height] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
                idx++
            }
        }

        val buffer = ByteBuffer.allocateDirect(floatArray.size * 4)
        buffer.order(ByteOrder.nativeOrder())
        val fb = buffer.asFloatBuffer()
        fb.put(floatArray)
        fb.rewind()
        return fb
    }

    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + kotlin.math.exp(-x))
    }

    fun close() {
        ortSession.close()
    }
}
